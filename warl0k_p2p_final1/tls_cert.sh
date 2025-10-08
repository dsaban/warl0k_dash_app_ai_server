# CA
openssl req -x509 -newkey rsa:2048 -days 365 -nodes -keyout ca.key -out ca.crt -subj "/CN=warlok-ca"
# Hub
openssl req -newkey rsa:2048 -nodes -keyout hub.key -out hub.csr -subj "/CN=hub"
openssl x509 -req -in hub.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out hub.crt -days 365
# Peer (shared for demo)
openssl req -newkey rsa:2048 -nodes -keyout peer.key -out peer.csr -subj "/CN=peer"
openssl x509 -req -in peer.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out peer.crt -days 365
