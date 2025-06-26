#!/bin/sh

bin ?= main
debug_model ?=
image_viewer ?= swayimg
output_image ?= target/output.png

all:
	cargo build --examples

debug:
	cargo run $(debug_model) --bin $(bin)

debug-show:
	cargo run $(debug_model) --bin $(bin) && if [ -f $(output_image) ]; then $(image_viewer) $(output_image); fi

example:
	cargo run --example $(bin)

example-show:
	cargo run --example $(bin) && if [ -f $(output_image) ]; then $(image_viewer) $(output_image); fi

nix-shell:
	nix-shell

clean:
	cargo clean
