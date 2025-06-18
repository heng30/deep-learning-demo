#!/bin/sh

bin ?= main
image_viewer ?= swayimg
output_image ?= target/output.png

all: bins

bins:
	cargo build --bins

debug:
	cargo run --bin $(bin)

debug-show:
	cargo run --bin $(bin) && if [ -f $(output_image) ]; then $(image_viewer) $(output_image); fi

realse:
	cargo build --release --bin $(bin)

nix-shell:
	nix-shell

clean:
	cargo clean
