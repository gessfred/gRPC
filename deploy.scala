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
    /*
--rnn_bptt_len 35  --rnn_clip 0.25  --rnn_use_pretrained_emb True  --rnn_tie_weights True  --rnn_weight_norm False  --manual_seed 6  --evaluate False  --summary_freq 100  --timestamp 1584012887_l2-0.0005_lr-0.01_epochs-90_batchsize-256_basebatchsize-None_num_mpi_process_1_n_sub_process-1_topology-complete_optim-local_sgd_comm_info-  --track_time False  --track_detailed_time False  --display_tracked_time False  --checkpoint ./data/checkpoint  --save_all_models False  --user lin  --project distributed_adam_type_algorithm  --backend mpi  --use_ipc False  --hostfile hostfile  --mpi_path $HOME/.openmpi  --python_path $HOME/conda/envs/pytorch-py3.6/bin/python  --num_workers 2  --n_mpi_process 1  --n_sub_process 1  --world 0,0  --on_cuda True  --comm_device cuda  --local_rank 0  --clean_python False
      \
      \
       \
        \
      \
        \
       \
      \
       \
        \
    --python_path $HOME/conda/envs/pytorch-py3.6/bin/python --mpi_path $HOME/.openmpi/
    */
    val tag = "tao"
    def run(rank: Int) = Command("/home/user/LocalSGD-Code/distributed_code", 
                    "main.py",
                    ("--arch",  "resnet20") ::
                    ("--local_rank", rank.toString) ::
                    ("--optimizer", "local_sgd") ::
                    ("--avg_model", "True") ::
                    ("--experiment", "demo") ::
                    ("--manual_seed", "6") ::
                    ("--data", "cifar10") ::
                    ("--pin_memory", "True") :: // DataLoader: if True, the data loader will copy Tensors into CUDA pinned memory 
                    ("--batch_size", "128") ::
                    ("--base_batch_size", "64") ::
                    ("--num_workers", "2") ::
                    ("--num_epochs", "0") :: // pytorch DataLoader arg. refer to https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
                    ("--partition_data", "random") ::
                    ("--reshuffle_per_epoch", "True") ::
                    ("--stop_criteria", "epoch") ::
                    ("--n_mpi_process", "2") ::
                    ("--n_sub_process", "1") ::
                    ("--world", "0,0") ::
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
|      resources:
|        limits:
|          nvidia.com/gpu: 1
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
|      - name: NCCL_DEBUG
|        value: INFO
|      - name: MASTER_ADDR
|        value: 192.168.0.6
|      - name: MASTER_PORT
|        value: "29500"
|      - name: NCCL_SOCKET_IFNAME
|        value: eth0
|      - name: NCCL_DEBUG_SUBSYS
|        value: NET
|      - name: NCCL_COMM_ID
|        value: 192.168.0.6:29500
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

/*
("--work_dir", ".") :: 
                    ("--data", "cifar10") ::
                    ("--arch", "resnet20") ::
                    ("--data_dir", "./data") ::
                    ("--lr", "0.01") ::
                    ("--lr_scheduler", "MultiStepLR") ::
                    ("--optimizer", "local_sgd") ::
                    ("-j", "2") ::
                    ("--world", "0,0") ::
                    ("--hostfile", "hostfile") ::
                    ("--local_rank", rank.toString) ::
                    ("--remote_exec", "False") :: //??
                    ("--use_lmdb_data", "False") :: //??
                    ("--pin_memory", "True") ::
                    ("--train_fast", "False") ::
                    ("--stop_criteria", "epoch") ::
                    ("--num_epochs", "90") ::
                    ("--num_iterations", "32000") ::
                    ("--avg_model", "False") ::
                    ("--reshuffle_per_epoch", "False") ::
                    ("--batch_size", "256") ::
                    ("--lr_decay", "0.01") ::
                    ("--lr_patience", "10") ::
                    ("--lr_scaleup", "False") ::
                    ("--lr_warmup", "False") ::
                    ("--lr_warmup_epochs_upper_bound", "150") ::
                    ("--adam_beta_1", "0.9") ::
                    ("--adam_beta_2", "0.999") ::
                    ("--adam_eps", "1e-08") ::
                    ("--graph_topology", "complete") ::
                    ("--compress_warmup_values", "0.75,0.9375,0.984375,0.996,0.999") :://??
                    ("--compress_warmup_epochs", "0") ::
                    ("--is_biased", "False") ::
                    ("--majority_vote", "False") ::
                    ("--consensus_stepsize", "0.9") ::
                    ("--evaluate_consensus", "False") ::
                    ("--mask_momentum", "False") ::
                    ("--clip_grad", "False") ::
                    ("--local_step", "1") ::
                    ("--turn_on_local_step_from", "0") ::
                    ("--momentum_factor", "0.9") ::
                    ("--use_nesterov", "False") ::
                    ("--weight_decay", "0.0005") ::
                    ("--drop_rate", "0.0") ::
                    ("--densenet_growth_rate", "12") ::
                    ("--densenet_bc_mode", "False") ::
                    ("--densenet_compression", "0.5") ::
                    ("--wideresnet_widen_factor", "4") ::
                    ("--rnn_n_hidden", "200") ::
                    ("--rnn_n_layers", "2") ::
*/