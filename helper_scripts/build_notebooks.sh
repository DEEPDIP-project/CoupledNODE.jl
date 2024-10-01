#!/bin/bash

echo "Building notebooks..."

# pre-commit passes staged files as arguments
FILENAMES=$(echo "$@" | grep 'examples/src/\.jl$')
echo "Files changed: $FILENAMES"

if [ -n "$FILENAMES" ]; then
    julia make_examples.jl $FILENAMES
else
    echo "No Julia notebook source files changed."
fi
