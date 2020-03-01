object Deploy extends App {
    def tab(x: Int) = 0.until(x).map(x => "  ").reduceLeft(_ + _)
    val env = Map("MASTER_ADDR" -> "192.168.0.4", "MASTER_PORT" -> "\"29500\"", "GLOO_SOCKET_IFNAME" -> "eth0")
    val nodes = "iccluster088" :: "iccluster095" :: Nil
    val cmd = "cat <<EOF | kubectl apply -f -\n"
    val eof = "\nEOF"
    case class Node(domain: String, rank: Int) {
        val name = if(rank == 0) "master" else s"slave${rank}"
    }
    case class Command(bin: String, args: List[String])
    //192.168.0.4
    def pod(node: Node): String = s"""apiVersion: v1
|kind: Pod
|metadata:
|    name: ${node.name}
|spec:
|    restartPolicy: Never
|    nodeName: ${node.domain}
|    volumes:
|     - name: datasets
|       hostPath:
|         path: /mnt/data
|     - name: mdb-creds
|       secret:
|         secretName: mongodb-secret
|         items:
|         - key: username
|           path: admin/username
|           mode: 0444
|         - key: password
|           path: admin/password
|           mode: 0444
|    containers:
|    - name: ${node.name}
|      image: gessfred/pyparsa:nccl
|      imagePullPolicy: Always
|      command: [ "python" ]
|      args: [ "/pyparsa/lib/mnist.py", "--lr", "0.01", "--dtype", "32bit", "--backend", "nccl" ]
|      resources:
|        limits:
|          nvidia.com/gpu: 1
|      ports:
|      - name: rendezvous
|        containerPort: 60000
|      volumeMounts:
|       - name: datasets
|         mountPath: /mnt/data
|       - name: mdb-creds
|         mountPath: /etc/mdb-creds
|         readOnly: true
|      env:
|      - name: MONGO_USR
|        value: /etc/mdb-creds/admin/username
|      - name: MONGO_PWD
|        value: /etc/mdb-creds/admin/password
|      - name: MASTER_ADDR
|        value: 192.168.0.6
|      - name: MASTER_PORT
|        value: "29500"
|      - name: GLOO_SOCKET_IFNAME
|        value: eth0
|      - name: DATAPATH
|        value: /mnt/data
|      - name: RANK
|        value: \"${node.rank}\"""".stripMargin
    val spec = nodes.zip(0.until(nodes.length)).map{
        case (node, rank) => Node(node, rank)
    }.map(pod).mkString("\n---\n")//pod(nodes.head, 0, "master")
    println((cmd+spec+eof))
}