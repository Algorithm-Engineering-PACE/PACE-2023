# PACE-2023
twin-width
## running
```
Process graph from stdin(PACE 2023) - cat <file_name> | python3 submission_main
Process graph from instance -  python3 solution.py "process-graph-from-instance" <file_name>
Process graph from directory - python3 solution.py "process-graphs-from-dir" <dict_name>
Clean results - python3 solution.py "clean-results"

use pace_verifier as follows -
// to create pace output (contraction_tree) run this command
python3 solution.py "process-graph-from-instance" <file_name>
// to verify output run this command (file_name* = file name without "gr" postfix)
python3 pace_verifier.py <file_name> <file_name*>_pace_output.gr

```

## Creating submission
In order to create a submission use a linux machine in the project folder run the following:

`tar -zcf submission_main.tgz *`.

Upload to optil.io `submission_main.tgz` file
