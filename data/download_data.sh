#!/usr/bin/env bash

# Name of the file that contains the URLs
FILE="datasets.txt"

# Check if the file exists
if [[ ! -f "$FILE" ]]; then
  echo "File '$FILE' not found!"
  exit 1
fi

# Read each line from the file
while IFS= read -r URL
do
  # Skip empty lines if any
  if [[ -n "$URL" ]]; then
    echo "Downloading: $URL"
    wget "$URL"
  fi
done < "$FILE"
