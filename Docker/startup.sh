#!/bin/bash
service nis start
service ssh start
su -P $USER -c "$@"
#source /o pt/miniconda3/bin/activate


# #!/bin/bash

# echo "PORT 30100" > /etc/ssh/sshd_config.d/customport.conf
# echo "PermitRootLogin no" >> /etc/ssh/sshd_config.d/customport.conf
# echo "AllowUsers baack $USER $ADDUSER" >> /etc/ssh/sshd_config.d/customport.conf

# service nis start
# service ssh start
# exec bash -i
