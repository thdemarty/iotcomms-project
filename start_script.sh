#!/bin/bash

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

LOG_FILE="${1:-log_${TIMESTAMP}.txt}"
DATA_FILE="${2:-data_${TIMESTAMP}.csv}"


echo "starting terminal"


script -q -c "make term" "$LOG_FILE"

echo "processing data"

echo "Timestamp,NodeID,Environment,RSSI,LQI" > "$DATA_FILE"

awk '
/\[DATA\]/ {
    timestamp = $1 " " $2
    gsub(",", ".", timestamp)
    gsub(",", "", $0)
    print timestamp "," $5 "," $6 "," $7 "," $8
}
' "$LOG_FILE" >> "$DATA_FILE"

echo "csv written"