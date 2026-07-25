use std::{
    env, fs,
    path::{Path, PathBuf},
};
use zerofps_assets::{MeshFormat, import_file};

fn main() {
    let roots: Vec<PathBuf> = env::args_os().skip(1).map(PathBuf::from).collect();
    let roots = if roots.is_empty() {
        vec![PathBuf::from("models")]
    } else {
        roots
    };
    let mut files = Vec::new();
    for root in roots {
        collect(&root, &mut files);
    }
    files.sort();

    let mut passed = 0;
    let mut failed = 0;
    let mut unsupported = 0;
    for path in files {
        if MeshFormat::from_path(&path).is_none() {
            println!("UNSUPPORTED\t{}", path.display());
            unsupported += 1;
            continue;
        }
        match import_file(&path) {
            Ok(asset) => {
                println!(
                    "PASS\t{}\tformat={}\taxis={}\tvertices={}\ttriangles={}\tprimitives={}\tmaterials={}\ttextures={}\twarnings={}",
                    path.display(),
                    asset.source.format,
                    asset.source.up_axis.label(),
                    asset.vertices.len(),
                    asset.triangle_count(),
                    asset.primitives.len(),
                    asset.materials.len(),
                    asset.textures.len(),
                    asset.warnings.len(),
                );
                for warning in asset.warnings {
                    println!("WARN\t{}\t{warning}", path.display());
                }
                passed += 1;
            }
            Err(error) => {
                println!("FAIL\t{}\t{error}", path.display());
                failed += 1;
            }
        }
    }
    println!("SUMMARY\tpassed={passed}\tfailed={failed}\tunsupported={unsupported}");
    if failed != 0 {
        std::process::exit(1);
    }
}

fn collect(path: &Path, output: &mut Vec<PathBuf>) {
    if path.is_file() {
        output.push(path.to_owned());
        return;
    }
    let Ok(entries) = fs::read_dir(path) else {
        eprintln!("WARN\t{}\tcannot read path", path.display());
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect(&path, output);
        } else if matches!(
            path.extension()
                .and_then(|extension| extension.to_str())
                .map(|extension| extension.to_ascii_lowercase())
                .as_deref(),
            Some("obj" | "ply" | "stl" | "gltf" | "glb")
        ) {
            output.push(path);
        }
    }
}
