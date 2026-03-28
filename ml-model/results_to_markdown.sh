#!/usr/bin/env bash
#
# ==================================================================
# This is a one-shot LLM file. I recommend against reading it.
# ==================================================================
#
# Convcerts the output of run_all to a markdown file

if [ $# -ne 1 ]; then
    echo "Usage: $0 <logfile>"
    exit 1
fi

in="$1"

awk '
function trim(s){ gsub(/^[ \t\r\n]+|[ \t\r\n]+$/, "", s); return s }
function collapse(s){ gsub(/[ \t]+/, " ", s); return s }

BEGIN {
  FS = "\n"
  section = 0
}

/^Running:/ {
  # emit previous section if any
  if (section) {
    process_section()
  }
  # start new section
  section = 1
  section_lines = 0
  delete lines
  lines[++section_lines] = $0
  next
}

{
  if (section) {
    lines[++section_lines] = $0
  }
}

END {
  if (section) process_section()
}

function process_section(   i, line, s, m, id, idpart, method_heading, images_prefix, table_start, rows, r, row) {
  s = ""; m = ""; id = ""
  # parse Running line for Scenario and Method numbers
  for (i=1;i<=section_lines;i++) {
    line = lines[i]
    if (line ~ /^Running:/) {
      # examples: "Running: CNN | Scenario 1 | Method 1"
      if (match(line, /Running:[[:space:]]*([a-zA-Z]+)/, a)) model = a[1]
      if (match(line, /Scenario[[:space:]]*([0-9]+)/, a)) s = a[1]
      if (match(line, /Method[[:space:]]*([0-9]+)/, a)) m = a[1]
    }
    # detect "Testing with env N" or "env N" to capture id
    if (line ~ /env[[:space:]]*[0-9]+/ || line ~ /id[[:space:]]*[0-9]+/) {
      if (match(line, /env[[:space:]]*([0-9]+)/, a)) id = a[1]
      else if (match(line, /id[[:space:]]*[:=]? *([0-9]+)/, a)) id = a[1]
    }
  }

  if (s=="") s="0"
  if (m=="") m="0"
  idpart = (id!="" ? "_id" id : "")

  # print headings and images
  if (s=="1" && m=="1") {
    printf "# %s\n\n", model
  }
  if (m=="1") {
    printf "## Scenario %s\n\n", s
  }
  if (id=="" || id=="0") {
    printf "### Method %s\n\n", m
  }
  if (id != "") {
    printf "#### Testing with id %s\n\n", id
  }
  printf "![training curve](saves/cnn_trainingcurve_s%s_m%s_f100_o50%s.png)\n", s, m, idpart
  printf "![confusion matrix](saves/cnn_confusionmatrix_s%s_m%s_f100_o50%s.png)\n\n", s, m, idpart

  # find table header start (line that contains "precision" and "recall")
  table_start = 0
  for (i=1;i<=section_lines;i++) {
    line = lines[i]
    if (line ~ /precision/ && line ~ /recall/) { table_start = i; break }
  }
  if (!table_start) {
    print ""   # no table in this section
    return
  }

  # collect table rows: following non-empty lines that look like rows (letters and numbers)
  rows = 0
  for (i=table_start+1;i<=section_lines;i++) {
    line = lines[i]
    # stop when there is an empty line followed by non-table content or end of section
    if (trim(line) == "" && (i+1>section_lines || trim(lines[i+1]) == "")) break
    # consider lines that have at least one word and one number -> table row
    if (line ~ /[A-Za-z]/ && line ~ /[0-9]/) {
      rows++
      row = collapse(trim(line))
      # normalize: replace multiple spaces with single; keep as one string
      table[rows] = row
    }
  }

  # print markdown table header (matches requested layout)
  printf "|              | precision |   recall |  f1-score |  support |\n"
  printf "| ------------ | --------- | -------- | --------- | -------- |\n"

  # convert each collected row into markdown row
  for (r=1;r<rows;r++) {
    row = table[r]
    # split by space
    n = split(row, parts, " ")
    # heuristic: last token is support, last-1 is f1-score, last-2 recall, last-3 precision, rest joined is label
    if (n==3 && parts[2] ~ /^[0-9]*\.?[0-9]+%?$/ && parts[3] ~ /^[0-9]+$/) {
      printf "|              |           |          |           |          |\n"
      label = parts[1]; prec = ""; recall = ""; f1 = parts[2]; support = parts[3]
    } else {
      support = (n>=1 ? parts[n] : "")
      f1 = (n>=2 ? parts[n-1] : "")
      recall = (n>=3 ? parts[n-2] : "")
      prec = (n>=4 ? parts[n-3] : "")
      label = ""
      for (j=1;j<n-3;j++) { label = label (j>1 ? " " : "") parts[j] }
    }
    label = sprintf("%12s", label)   # pad label to align visually (not required)
    # empty numeric fields may be "-" in some logs; keep as-is
    printf "| %s | %8s  | %8s | %8s  | %8s |\n", label, prec, recall, f1, support
  }

  # blank line after table
  printf "\n"

  # clear table array for next section
  delete table
}
' "$in"
