set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]



build:
  @cd src/plugin
  cargo build --manifest-path src/plugin/Cargo.toml --release --target wasm32-unknown-unknown 
  @cp src/plugin/target/wasm32-unknown-unknown/release/plugin.wasm src



prepare:
  rustup target add wasm32-unknown-unknown


hello:
  #!python
  import os
  print("hello {}".format(os.sys))
  