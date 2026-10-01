#!/usr/bin/env nu

# Download model weights if not already present
def main [] {
  let target_dir = "src/weights"
  let target_file = ($target_dir | path join "encodec_24khz.safetensors")
  let url = "https://huggingface.co/facebook/encodec_24khz/resolve/main/model.safetensors"

  mkdir $target_dir

  if ($target_file | path exists) {
    print $"($target_file) already exists. Skipping."
  } else {
    print $"Downloading ($target_file)..."
    http get --raw $url | save --raw $target_file
    print "Done."
  }
}
