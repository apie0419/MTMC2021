#!/bin/bash
# Show username after each process in nvidia-smi
# like:
# ...
# +------------------------------------------------------+
# | Processes:                                GPU Memory |
# |  GPU       PID  Type  Process name        Usage      |
# |======================================================|
# |    0    150752    C   python                  830MiB | User: user1
# |    1      2185    C   /usr/bin/python        1090MiB | User: user2
# ..

proc_lines=$(nvidia-smi | egrep "MiB \|$")
users=$(echo "$proc_lines" | awk '{print $3}' | xargs -n 1 ps -o user:20 --no-header -p)
awk 'BEGIN {s=-999}                                                                                                                                                NR==FNR {u[NR]=$1; next}
     { if ($3=="GPU")
         s = 0;
       if (u[s-2])
           {print $0 " User: " u[s-2]}
       else
           print $0;                                                                                                                                                 s=s+1
     }' <(echo "$users") <(nvidia-smi)