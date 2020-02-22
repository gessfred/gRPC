object Deploy extends App {
    def tab(x: Int) = 0.until(x).map(x => "  ").reduceLeft(_ + _)
    val env = Map("MASTER_ADDR" -> "192.168.0.4", "MASTER_PORT" -> "\"29500\"", "GLOO_SOCKET_IFNAME" -> "eth0")
    val nodes = "iccluster088" :: "iccluster095" :: Nil
    val cmd = "cat <<EOF | kubectl apply -f -\n"
    val eof = "\nEOF"
    case class Node(domain: String, rank: Int) {
        val name = if(rank == 0) "master" else s"slave${rank}"
    }
    val volume = """apiVersion: v1
|kind: PersistentVolume
|metadata:
|  name: task-pv-volume
|  labels:
|    type: local
|spec:
|  storageClassName: manual
|  accessModes:
|   - ReadWriteOnce
|  hostPath:
|    path: "/mnt/data"""".stripMargin
    def pod(node: Node): String = s"""apiVersion: v1
|kind: Pod
|metadata:
|    name: ${node.name}
|spec:
|    nodeName: ${node.domain}
|    volumes:
|     - name: datasets
|       hostPath:
|         path: /mnt/data
|    containers:
|    - name: ${node.name}
|      image: gessfred/pyparsa
|      command: [ "python" ]
|      args: [ "/jet/lib/mnist.py", "--lr", "0.01" ]
|      volumeMounts:
|       - name: datasets
|         mountPath: /mnt/data
|      env:
|      - name: MASTER_ADDR
|        value: 192.168.0.4
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