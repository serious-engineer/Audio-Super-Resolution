# DRT (Diagnostic Rhyme Test) Template

1. Prepare a list of 96 minimal pairs (e.g., 'veal' vs 'feel', 'bat' vs 'pat').
2. Play either reference (WB) or processed (model/baseline) audio for a word and ask the listener to choose which of the pair was heard.
3. Randomize order, collect responses to CSV: listener_id, file, correct, system.
4. Compute % correct per system and do a McNemar's test for significance.
