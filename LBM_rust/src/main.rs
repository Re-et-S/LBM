mod midi;
use midi::{MidiFile, events_to_chunks};

mod bpe;
use bpe::{run_clustering_pipeline, tokenize_data, run_bpe, generate_sequence, decode_sequence};

use walkdir::WalkDir;
use indicatif::{ProgressBar, ProgressStyle};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use serde::Serialize;

#[derive(Serialize)]
struct FileMetadata {
    filename: String,
    chunk_count: usize,
    start_offset_bytes: u64,
}

fn process_single_file<W: Write>(path: &Path, writer: &mut W) -> Result<usize, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    // Buffering locally for read is good
    let mut reader = std::io::BufReader::new(file);

    // Parsing might fail (Corrupt header, truncated file)
    let midi_file = MidiFile::parse(&mut reader)?;
    
    // Heuristic 1: Skip files with weird time divisions (SMPTE)
    // (Already handled in your parse function, but good to remember)

    let flattened = midi_file.flatten();
    
    // Heuristic 2: Skip empty files
    if flattened.is_empty() {
        return Ok(0);
    }

    // Heuristic 3: Skip files that are too short (e.g. < 1 second)
    // You can check the timestamp of the last event in 'flattened'
    let last_tick = flattened.last().unwrap().abs_ticks;
    if last_tick < midi_file.division as u64 { // Less than 1 beat?
         return Ok(0); 
    }

    let chunks = events_to_chunks(&flattened, midi_file.division);

    for chunk in &chunks {
        writer.write_all(&chunk.to_binary())?;
    }

    Ok(chunks.len())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input_dir = "bach_midi_files";
    let output_bin = "dataset.bin";
    let output_index = "dataset_index.json";

    // 1. Setup Outputs
    let mut bin_writer = BufWriter::new(File::create(output_bin)?);
    let mut metadata_list = Vec::new();
    let mut current_byte_offset = 0u64;

    // 2. Collect files
    let midi_files: Vec<PathBuf> = WalkDir::new(input_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "mid"))
        .map(|e| e.path().to_owned())
        .collect();

    println!("Found {} MIDI files. Processing...", midi_files.len());

    let pb = ProgressBar::new(midi_files.len() as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
        .unwrap());

    for path in midi_files {
        // Run processing in a separate function to catch errors easily
        match process_single_file(&path, &mut bin_writer) {
            Ok(count) => {
                if count > 0 {
                    // Only record metadata if we actually got chunks
                    metadata_list.push(FileMetadata {
                        filename: path.file_name().unwrap().to_string_lossy().into_owned(),
                        chunk_count: count,
                        start_offset_bytes: current_byte_offset,
                    });
                    current_byte_offset += (count * 36) as u64; // 36 bytes per chunk
                }
            },
            Err(e) => {
                // Log error but continue!
                pb.println(format!("Skipping {:?}: {}", path.file_name().unwrap(), e));
            }
        }
        pb.inc(1);
    }

    pb.finish_with_message("Done!");
    let index_file = File::create(output_index)?;
    serde_json::to_writer_pretty(index_file, &metadata_list)?;
    
    println!("Saved dataset to {} and index to {}", output_bin, output_index);

    let rawData = run_clustering_pipeline(output_bin)?;
    let (tokenized_data, vocab) = tokenize_data(&rawData);
    let (compressed_data, merge_rule) = run_bpe(tokenized_data, vocab.len());

    let sampled_sequence = generate_sequence(&compressed_data, 1000);
    let decoded_chunks = decode_sequence(&sampled_sequence, &vocab, &merge_rule);
    let bpe_out = "bpe_output_sample.bin";
    let mut bin_writer_bpe = BufWriter::new(File::create(bpe_out)?);

    for chunk in decoded_chunks {
        bin_writer_bpe.write_all(&chunk)?;
    }
    println!("Done! Saved to {}", bpe_out);
    Ok(())
}
