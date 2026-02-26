/// Build script: generates instruction methods from tables in gen/.

#[path = "gen/mod.rs"]
mod gen;

fn main() {
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let dest_path = std::path::Path::new(&out_dir).join("mnemonics.rs");
    let mut file = std::fs::File::create(&dest_path).unwrap();

    let mut all_tables: Vec<&[gen::Insn]> = Vec::new();
    all_tables.extend(gen::instructions::all_tables());
    all_tables.extend(gen::avx512::all_tables());

    gen::codegen::generate(&mut file, &all_tables).unwrap();

    // Rerun if gen/ sources change
    println!("cargo:rerun-if-changed=gen/mod.rs");
    println!("cargo:rerun-if-changed=gen/instructions.rs");
    println!("cargo:rerun-if-changed=gen/avx512.rs");
    println!("cargo:rerun-if-changed=gen/codegen.rs");
}
