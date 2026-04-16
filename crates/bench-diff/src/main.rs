//! Criterion baseline diff tool.
//!
//! After running `cargo bench -- --save-baseline <tag>`, criterion stores raw
//! samples under `<target>/criterion/<group>/<bench>/<tag>/estimates.json`.
//! This tool walks every such group, computes `new/mean - baseline/mean`,
//! and fails the process with a non-zero exit code if any group has
//! regressed beyond `--threshold` (default 5%).
//!
//! Usage:
//!
//!   bench-diff --baseline v0.1.0 [--threshold 0.05] [--criterion-dir PATH]
//!
//! No external deps — hand-rolled JSON parsing for the two fields we need.

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    process::ExitCode,
};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();

    let mut baseline: Option<String> = None;
    let mut threshold_pct: f64 = 5.0;
    let mut criterion_dir: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--baseline" => {
                i += 1;
                baseline = args.get(i).cloned();
            }
            "--threshold" => {
                i += 1;
                threshold_pct = args
                    .get(i)
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(5.0);
            }
            "--criterion-dir" => {
                i += 1;
                criterion_dir = args.get(i).map(PathBuf::from);
            }
            "-h" | "--help" => {
                eprintln!(
                    "usage: bench-diff --baseline <tag> [--threshold PCT] [--criterion-dir PATH]"
                );
                return ExitCode::from(0);
            }
            other => {
                eprintln!("unknown arg: {other}");
                return ExitCode::from(2);
            }
        }
        i += 1;
    }

    let Some(baseline) = baseline else {
        eprintln!("error: --baseline <tag> is required");
        return ExitCode::from(2);
    };

    let dir = match criterion_dir {
        Some(p) => p,
        None => match locate_criterion_dir() {
            Some(p) => p,
            None => {
                eprintln!("error: unable to locate <target>/criterion/. Pass --criterion-dir.");
                return ExitCode::from(2);
            }
        },
    };

    // Gather (group_name → (new_ns, base_ns)).
    let mut rows: BTreeMap<String, (Option<f64>, Option<f64>)> = BTreeMap::new();
    walk_estimates(&dir, &dir, &baseline, &mut rows);

    if rows.is_empty() {
        eprintln!(
            "error: no estimates found under {}. Did you run `cargo bench -- --save-baseline \
             {baseline}` first?",
            dir.display()
        );
        return ExitCode::from(2);
    }

    // Emit a table.
    println!();
    println!(
        "================ Baseline diff (baseline = {baseline}, threshold = {threshold_pct:.1}%) \
         ================"
    );
    println!(
        "{:<50} {:>14} {:>14} {:>8}  status",
        "bench", "baseline ns", "current ns", "delta%"
    );
    let mut regressions = 0usize;
    let mut improvements = 0usize;
    let mut seen_pairs = 0usize;
    for (name, (new_ns, base_ns)) in &rows {
        match (new_ns, base_ns) {
            (Some(n), Some(b)) => {
                seen_pairs += 1;
                let delta = (n - b) / b * 100.0;
                let status = if delta > threshold_pct {
                    regressions += 1;
                    "REGRESSION"
                } else if delta < -threshold_pct {
                    improvements += 1;
                    "improved"
                } else {
                    "ok"
                };
                println!(
                    "{:<50} {:>14.0} {:>14.0} {:>+8.2}  {}",
                    name, b, n, delta, status
                );
            }
            (Some(_), None) => {
                println!("{:<50} {:>14} {:>14} {:>8}  new bench", name, "-", "?", "-")
            }
            (None, Some(_)) => println!(
                "{:<50} {:>14} {:>14} {:>8}  missing run",
                name, "?", "-", "-"
            ),
            _ => {}
        }
    }
    println!(
        "==============================================================================================="
    );
    println!("summary: {seen_pairs} compared, {regressions} regressed, {improvements} improved");

    if regressions > 0 {
        ExitCode::from(1)
    } else {
        ExitCode::from(0)
    }
}

fn locate_criterion_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("CARGO_TARGET_DIR") {
        let p = PathBuf::from(dir).join("criterion");
        if p.is_dir() {
            return Some(p);
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        let mut p = exe.as_path();
        for _ in 0..6 {
            if let Some(parent) = p.parent() {
                let candidate = parent.join("criterion");
                if candidate.is_dir() {
                    return Some(candidate);
                }
                p = parent;
            } else {
                break;
            }
        }
    }
    let fallback = PathBuf::from("target/criterion");
    if fallback.is_dir() {
        Some(fallback)
    } else {
        None
    }
}

/// Recursively walk the criterion tree collecting `<group>/<bench>/<side>/estimates.json`.
fn walk_estimates(
    root: &Path,
    cur: &Path,
    baseline: &str,
    out: &mut BTreeMap<String, (Option<f64>, Option<f64>)>,
) {
    let Ok(entries) = std::fs::read_dir(cur) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            // Recurse — but skip special subtrees that don't hold benches.
            let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if name == "report" {
                continue;
            }
            walk_estimates(root, &p, baseline, out);
        } else if p.file_name().and_then(|s| s.to_str()) == Some("estimates.json") {
            // Path shape: <root>/<...group path.../>/<side>/estimates.json
            let side_dir = p.parent().expect("parent");
            let bench_id_dir = side_dir.parent().expect("parent^2");
            let side = side_dir.file_name().and_then(|s| s.to_str()).unwrap_or("");
            // Only consider `new` and the named baseline; ignore `base`/`change`.
            let bucket = if side == "new" {
                Bucket::New
            } else if side == baseline {
                Bucket::Base
            } else {
                continue;
            };
            let rel = bench_id_dir
                .strip_prefix(root)
                .unwrap_or(bench_id_dir)
                .to_string_lossy()
                .into_owned();
            let mean = read_mean_ns(&p);
            let entry = out.entry(rel).or_insert((None, None));
            match bucket {
                Bucket::New => entry.0 = mean,
                Bucket::Base => entry.1 = mean,
            }
        }
    }
}

enum Bucket {
    New,
    Base,
}

fn read_mean_ns(path: &Path) -> Option<f64> {
    let text = std::fs::read_to_string(path).ok()?;
    let mean_idx = text.find("\"mean\"")?;
    let rest = &text[mean_idx..];
    let pe_idx = rest.find("\"point_estimate\"")?;
    let after = &rest[pe_idx + "\"point_estimate\"".len()..];
    let colon = after.find(':')?;
    let num_start = colon + 1;
    let num_end = after[num_start..]
        .find(|c: char| {
            c != '.' && c != '-' && c != '+' && c != 'e' && c != 'E' && !c.is_ascii_digit()
        })
        .unwrap_or(after.len() - num_start);
    after[num_start..num_start + num_end].trim().parse().ok()
}
