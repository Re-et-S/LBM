mod bpe;

use std::fs::File;
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    
    if args.len() < 2 {
        eprintln!("Usage: {} <sequence_path> [vocab_path]", args[0]);
        std::process::exit(1);
    }

    let sequence_path = &args[1];
    let vocab_path = if args.len() > 2 { &args[2] } else { "vocab.bin" };
    
    println!("Decoding sequence from {} using vocab {}", sequence_path, vocab_path);
    
    let (vocab, merge_rules) = bpe::load_vocab(vocab_path)?;
    let sequence = bpe::load_sequence(sequence_path)?;

    let decoded_chunks = bpe::decode_sequence(&sequence, &vocab, &merge_rules);

    println!("Successfully decoded sequence of length {} into {} base chunks.", sequence.len(), decoded_chunks.len());

    let out_path = format!("{}_decoded.bin", sequence_path);
    let mut out = File::create(&out_path)?;
    for chunk in &decoded_chunks {
        out.write_all(chunk)?;
    }
    println!("Saved decoded chunks to {}.", out_path);
    
    Ok(())
}
