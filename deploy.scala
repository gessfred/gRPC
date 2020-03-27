import sys.process._
import scala.language.postfixOps

object Deploy extends App {
    def tab(x: Int) = 0.until(x).map(x => "  ").reduceLeft(_ + _)
    val env = Map("MASTER_ADDR" -> "192.168.0.4", "MASTER_PORT" -> "\"29500\"", "GLOO_SOCKET_IFNAME" -> "eth0")
    val nodes = "iccluster088" :: "iccluster095" :: Nil
    val cmd = "cat <<EOF | kubectl apply -f -\n"
    val eof = "\nEOF"
    case class Node(domain: String, rank: Int) {
        val name = if(rank == 0) "master" else s"slave${rank}"
    }
    case class Command(workDir: String, bin: String, args: List[(String, String)]) {
        def str(tabs: Int): String = {
            val tab = 0.until(tabs).map(i => "  ").mkString("")
            val sep ="\""
            val argsStr = args
                .flatMap{ case (key, value) => key :: value :: Nil}
                .map{arg:String => s"$sep$arg$sep"}.mkString(", ")
            s"""|${tab}workingDir: $workDir
            |${tab}command: [ "python" ]
            |${tab}args: [ $sep$bin$sep, ${argsStr} ]""".stripMargin
        }
    }
    
    val br: String = "git branch" !!

    val branch = br.split("\n").filter(_.contains("*"))(0).replaceAll("\\W", "")
    val cm: String = "git log --oneline"!!

    val commit = cm.split("\n").head.split(" ").head

    val tag = "tao"
    def run(rank: Int) = Command("/home/user/LocalSGD-Code/distributed_code", 
                    "main.py",
                    ("--arch",  "resnet50") ::
                    ("--local_rank", rank.toString) ::
                    ("--optimizer", "local_ef_sgd") ::
                    ("--avg_model", "True") ::
                    ("--experiment", "demo") ::
                    ("--manual_seed", "6") ::
                    ("--data", "cifar100") ::
                    ("--pin_memory", "True") :: // DataLoader: if True, the data loader will copy Tensors into CUDA pinned memory 
                    ("--batch_size", "128") ::
                    ("--base_batch_size", "64") ::
                    ("--num_workers", "4") ::
                    ("--num_epochs", "1") :: // pytorch DataLoader arg. refer to https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
                    ("--partition_data", "random") ::
                    ("--reshuffle_per_epoch", "True") ::
                    ("--stop_criteria", "epoch") ::
                    ("--n_mpi_process", "2") ::
                    ("--n_sub_process", "2") ::
                    ("--compress_width", "1") ::
                    ("--world", "0,1,0,1") ::
                    ("--on_cuda", "True") ::
                    ("--use_ipc", "False") ::
                    ("--lr", "0.1") ::
                    ("--lr_scaleup", "True") ::
                    ("--lr_warmup", "True") ::
                    ("--lr_warmup_epochs", "5") ::
                    ("--lr_scheduler", "MultiStepLR") ::
                    ("--lr_decay", "0.1") ::
                    ("--lr_milestones", "150,225") ::
                    ("--local_step", "32") ::
                    ("--turn_on_local_step_from", "150") ::
                    ("--backend", "nccl") ::
                    ("--weight_decay", "1e-4") ::
                    ("--use_nesterov", "True") ::
                    ("--momentum_factor", "0.9") ::
                    ("--hostfile", "hostfile") ::
                    ("--graph_topology", "complete") ::
                    ("--track_time", "True") ::
                    ("--display_tracked_time", "True") ::
                    Nil
                ).str(3).stripMargin

    //"/pyparsa/lib/mnist.py", "--lr", "0.01", "--dtype", "32bit", "--backend", "nccl"
    //192.168.0.4
    def pod(node: Node): String = s"""apiVersion: v1
|kind: Pod
|metadata:
|    name: ${node.name}
|    labels:
|      app: local-sgd-${node.rank+1}
|spec:
|    securityContext:
|      fsGroup: 1000
|    imagePullSecrets:
|    - name: regcred
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
|      image: gessfred/pyparsa:${tag}
|      imagePullPolicy: Always
${run(node.rank)}
|      ports:
|      - name: rendezvous
|        containerPort: 60000
|      - containerPort: 22
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
|      - name: NCCL_SOCKET_IFNAME
|        value: eth0
|      - name: NCCL_COMM_ID
|        value: 192.168.0.6:29500
|      - name: VCS_BRANCH
|        value: $branch
|      - name: VCS_COMMIT
|        value: $commit
|      - name: DATAPATH
|        value: /mnt/data
|      - name: RANK
|        value: \"${node.rank}\"""".stripMargin
    val spec = nodes.zipWithIndex.map{
        case (node, rank) => Node(node, rank)
    }.map(pod).mkString("\n---\n")//pod(nodes.head, 0, "master")
    println((cmd+spec+eof))

}

