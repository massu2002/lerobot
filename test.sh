ROOT=/home/data01/smolvla

find "$ROOT" -path '*/meta/info.json' -print0 \
  | while IFS= read -r -d '' file; do
      mapfile -t cams < <(
        jq -r '.features | keys[] | select(startswith("observation.images."))' "$file" \
        | sort -u
      )
      n=${#cams[@]}
      (( n < 2 )) && continue

      for ((i=0; i<n; i++)); do
        for ((j=i+1; j<n; j++)); do
          printf '%s,%s,%s\n' "${cams[i]}" "${cams[j]}" "$file"
        done
      done
    done \
  | sort \
  | awk -F',' '
      {
        pair = $1","$2
        files[pair] = files[pair] "\n  " $3
        count[pair]++
      }
      END {
        for (pair in files) {
          print "=== " pair " (count=" count[pair] ") ==="
          printf "%s\n\n", files[pair]
        }
      }'
