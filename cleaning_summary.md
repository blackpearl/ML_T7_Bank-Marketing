# Data Cleaning Summary: `bank-additional-full-cleaned.csv`

## Overview
Original file: `bank-additional-full.csv`  
Cleaned file: `bank-additional-full-cleaned.csv`  
Cleaning completed on: [Date here]

## Cleaning Steps
- Removed rows where `job == 'unknown'` (330 rows)
- Removed rows where `marital == 'unknown'` (80 rows)
- Replaced `'unknown'` in `education` (1731 entries) with mode value
- Replaced `'unknown'` in `default` (8597 entries) with `'Default_Unknown'`

## Impact
- Original row count: 41,188  
- Cleaned row count: 40,787  
- Total rows removed: **401**
- `'unknown'` values are now fully cleaned or clarified

## Notes
- Cleaned file matches original delimiter (`;`)
- `bank-additional-full.csv` excluded from Git tracking using `.gitignore`
