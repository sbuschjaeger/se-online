#!/bin/bash
echo "PORT $SSH_PORT" > /etc/ssh/sshd_config.d/customport.conf
service nis start
service ssh start
su -P $USER -c "$@"
#source /o pt/miniconda3/bin/activate
#echo "PermitRootLogin no" >> /etc/ssh/sshd_config.d/customport.conf
#echo "AllowUsers baack $USER $ADDUSER" >> /etc/ssh/sshd_config.d/customport.conf


# #!/bin/bash


# service nis start
# service ssh start
# exec bash -i
