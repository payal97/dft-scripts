#!/bin/bash

find "$PWD" -type f -name "AECCAR0" \
    -exec dirname "{}" \; \
    | xargs -I {} bash -c "cd '{}' && pwd && chgsum.pl AECCAR0 AECCAR2 && bader CHGCAR -ref CHGCAR_sum && ~/split_dos"
