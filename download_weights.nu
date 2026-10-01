#!/usr/bin/env nu

# Download model weights if not already present
def main [] {
  let target_dir = "src/weights"
  let target_file = ($target_dir | path join "encodec_24khz.safetensors")

  # Pin to specific commit revision instead of 'main' to prevent unexpected upstream changes
  let revision = "013b631c360814d9b23f2ec491f24d45863d0ca1"
  let url = $"https://huggingface.co/facebook/encodec_24khz/resolve/($revision)/model.safetensors"

  mkdir $target_dir

  if ($target_file | path exists) {
    print $"($target_file) already exists. Skipping."
  } else {
    print $"Downloading ($target_file) from revision ($revision)..."
    http get --raw $url | save --raw $target_file
    print "Done."
  }
}
