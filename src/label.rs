use std::collections::HashMap;

use crate::code_array::LabelMode;
use crate::error::{Error, Result};

/// A unique identifier for a label.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LabelId(pub(crate) u32);

impl core::fmt::Display for LabelId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "L{}", self.0)
    }
}

/// Jump type for labels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JmpType {
    /// Short jump (8-bit displacement).
    Short,
    /// Near jump (32-bit displacement).
    Near,
    /// Auto-select (short if possible).
    Auto,
}

/// A forward reference to an undefined label.
#[derive(Clone, Debug)]
pub(crate) struct JmpLabel {
    /// Offset from top to end of jump instruction.
    pub end_of_jmp: usize,
    /// Size of the jump address field (1 or 4).
    pub jmp_size: u8,
    /// Label mode for resolution.
    pub mode: LabelMode,
    /// Additional displacement.
    pub disp: i64,
}

/// A label handle returned to the user.
#[derive(Clone, Copy, Debug)]
pub struct Label {
    id: LabelId,
}

impl Label {
    pub fn id(&self) -> LabelId { self.id }
}

/// Definition of a label (offset into code buffer).
#[derive(Clone, Debug)]
struct LabelDef {
    offset: usize,
}

/// Manages both named (string) labels and anonymous (Label object) labels.
pub struct LabelManager {
    next_id: u32,
    /// Definitions: label_id → offset.
    defs: HashMap<LabelId, LabelDef>,
    /// Undefined forward references: label_id → list of JmpLabel.
    undefs: HashMap<LabelId, Vec<JmpLabel>>,
    /// Named label → label_id mapping (global scope).
    global_names: HashMap<String, LabelId>,
    /// Local label scopes (stack of name → label_id maps).
    local_scopes: Vec<HashMap<String, LabelId>>,
    /// Named label forward references.
    named_undefs: HashMap<String, Vec<(LabelId, JmpLabel)>>,
}

impl LabelManager {
    pub fn new() -> Self {
        Self {
            next_id: 1,
            defs: HashMap::new(),
            undefs: HashMap::new(),
            global_names: HashMap::new(),
            local_scopes: vec![HashMap::new()],
            named_undefs: HashMap::new(),
        }
    }

    /// Reset the label manager.
    pub fn reset(&mut self) {
        self.next_id = 1;
        self.defs.clear();
        self.undefs.clear();
        self.global_names.clear();
        self.local_scopes.clear();
        self.local_scopes.push(HashMap::new());
        self.named_undefs.clear();
    }

    /// Create a new anonymous label.
    pub fn create_label(&mut self) -> Label {
        let id = LabelId(self.next_id);
        self.next_id += 1;
        Label { id }
    }

    /// Define a label at the given code offset.
    pub fn define_label(&mut self, label: &Label, offset: usize) -> Result<()> {
        self.define_label_id(label.id, offset)
    }

    /// Define a label by ID at the given code offset.
    fn define_label_id(&mut self, id: LabelId, offset: usize) -> Result<()> {
        if self.defs.contains_key(&id) {
            return Err(Error::LabelIsRedefined);
        }
        self.defs.insert(id, LabelDef { offset });
        Ok(())
    }

    /// Get the offset for a defined label.
    pub fn get_offset(&self, label: &Label) -> Option<usize> {
        self.defs.get(&label.id).map(|d| d.offset)
    }

    /// Get the offset for a label by ID.
    pub fn get_offset_by_id(&self, id: LabelId) -> Option<usize> {
        self.defs.get(&id).map(|d| d.offset)
    }

    /// Add an undefined forward reference.
    pub(crate) fn add_undef(&mut self, id: LabelId, jmp: JmpLabel) {
        self.undefs.entry(id).or_default().push(jmp);
    }

    /// Resolve all forward references for a label that was just defined.
    ///
    /// Returns a list of (patch_offset, value, size) for each forward ref.
    pub fn resolve_label(
        &mut self,
        id: LabelId,
        label_offset: usize,
        _is_auto_grow: bool,
    ) -> Result<Vec<(usize, u64, u8, LabelMode)>> {
        let refs = match self.undefs.remove(&id) {
            Some(refs) => refs,
            None => return Ok(Vec::new()),
        };

        let mut patches = Vec::new();
        for jmp in refs {
            let patch_offset = jmp.end_of_jmp - jmp.jmp_size as usize;
            let disp = match jmp.mode {
                LabelMode::AddTop => label_offset as u64,
                LabelMode::Abs => {
                    // Will be resolved later with actual address
                    0u64
                }
                LabelMode::AsIs => {
                    let d = label_offset as i64 - jmp.end_of_jmp as i64 + jmp.disp;
                    if jmp.jmp_size <= 4 {
                        if !(i32::MIN as i64..=i32::MAX as i64).contains(&d) {
                            return Err(Error::OffsetIsTooBig);
                        }
                    }
                    if jmp.jmp_size == 1 {
                        if !(-128..=127).contains(&d) {
                            return Err(Error::LabelIsTooFar);
                        }
                    }
                    d as u64
                }
            };

            let final_disp = if jmp.mode != LabelMode::AsIs {
                disp.wrapping_add(jmp.disp as u64)
            } else {
                disp
            };

            patches.push((patch_offset, final_disp, jmp.jmp_size, jmp.mode));
        }
        Ok(patches)
    }

    /// Define a named label.
    pub fn define_named_label(&mut self, name: &str, offset: usize) -> Result<LabelId> {
        if name == "@b" || name == "@f" {
            return Err(Error::BadLabelStr);
        }

        // Handle @@ (anonymous backward/forward) labels
        if name == "@@" {
            // Remove any existing @f definition
            self.global_names.remove("@f");
            let id = LabelId(self.next_id);
            self.next_id += 1;
            self.global_names.insert("@b".to_string(), id);
            self.global_names.insert("@f".to_string(), id);
            self.define_label_id(id, offset)?;
            return Ok(id);
        }

        // Check if it's a local label (starts with '.')
        let is_local = name.starts_with('.');
        let scope = if is_local {
            self.local_scopes.last_mut().unwrap()
        } else {
            &mut self.global_names
        };

        if scope.contains_key(name) {
            return Err(Error::LabelIsRedefined);
        }

        let id = LabelId(self.next_id);
        self.next_id += 1;
        scope.insert(name.to_string(), id);
        self.define_label_id(id, offset)?;
        Ok(id)
    }

    /// Look up a named label's ID.
    pub fn find_named_label(&self, name: &str) -> Option<LabelId> {
        // Check local scopes first (back to front)
        if name.starts_with('.') {
            for scope in self.local_scopes.iter().rev() {
                if let Some(&id) = scope.get(name) {
                    return Some(id);
                }
            }
        }
        self.global_names.get(name).copied()
    }

    /// Enter a local label scope.
    pub fn enter_local(&mut self) {
        self.local_scopes.push(HashMap::new());
    }

    /// Leave a local label scope.
    pub fn leave_local(&mut self) -> Result<()> {
        if self.local_scopes.len() <= 1 {
            return Err(Error::UnderLocalLabel);
        }
        self.local_scopes.pop();
        Ok(())
    }

    /// Check if there are undefined labels.
    pub fn has_undef_labels(&self) -> bool {
        !self.undefs.is_empty()
    }
}

impl Default for LabelManager {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_define() {
        let mut mgr = LabelManager::new();
        let label = mgr.create_label();
        mgr.define_label(&label, 0x100).unwrap();
        assert_eq!(mgr.get_offset(&label), Some(0x100));
    }

    #[test]
    fn test_redefined_label() {
        let mut mgr = LabelManager::new();
        let label = mgr.create_label();
        mgr.define_label(&label, 0x100).unwrap();
        assert!(mgr.define_label(&label, 0x200).is_err());
    }

    #[test]
    fn test_named_labels() {
        let mut mgr = LabelManager::new();
        let id = mgr.define_named_label("my_func", 0x50).unwrap();
        assert_eq!(mgr.find_named_label("my_func"), Some(id));
    }

    #[test]
    fn test_local_scope() {
        let mut mgr = LabelManager::new();
        mgr.enter_local();
        mgr.define_named_label(".loop", 0x10).unwrap();
        assert!(mgr.find_named_label(".loop").is_some());
        mgr.leave_local().unwrap();
        // After leaving scope, the local label is gone from search
    }
}
