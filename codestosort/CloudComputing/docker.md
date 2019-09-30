# docker

> a vm in a vm

## installation

The commands: copied from [docker.com](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
```
sudo apt-get remove docker docker-engine docker.io containerd runc
sudo apt-get update
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo apt-key fingerprint 0EBFCD88
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"

sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io

sudo docker run hello-world
```

### some minor configurations
* if you see `localhost not found`  modify /etc/hosts and change `127.0.0.1   localhost` to `127.0.0.1   <username>   localhost`
* to use docker without sudo: `sudo usermod -a -G docker <user>`, then reboot (`sudo reboot`)

### usages

* list container `docker container ls`
* stop container `docker stop <id or name>`
* remove container `docker container rm <id or name>...` 

same for network images ... etc  

* enter into docker: `docker exec -it hadoop-master bash`

## basic example: Wordpress

> use stack.yml for complicated docker creation

following https://hub.docker.com/_/wordpress/

pull wordpress docker image with `docker pull Wordpress`
stack.yml:
```
version: '3.1'

services:

  wordpress:
    image: wordpress
    restart: always
    ports:
      - 8080:80
    environment:
      WORDPRESS_DB_HOST: db
      WORDPRESS_DB_USER: exampleuser
      WORDPRESS_DB_PASSWORD: examplepass
      WORDPRESS_DB_NAME: exampledb

  db:
    image: mysql:5.7
    restart: always
    environment:
      MYSQL_DATABASE: exampledb
      MYSQL_USER: exampleuser
      MYSQL_PASSWORD: examplepass
      MYSQL_RANDOM_ROOT_PASSWORD: '1'
```

create docker container by `docker stack deploy -c stack.yml my_wordpress`


### References
[install docker ce for ubuntu](https://docs.docker.com/install/linux/docker-ce/ubuntu/)