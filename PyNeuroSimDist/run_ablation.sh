#!/bin/bash
###
 # @Author: Wang Zongwu
 # @Date: 2024-02-28 22:40:25
 # @LastEditTime: 2024-04-12 02:55:11
 # @LastEditors: Wang Zongwu
 # @Description: MICRO 2024 parallel run script
 # @FilePath: /ReRAM_SNN_Acce/PyNeuroSim/run.sh
 # @Mail: wangzongwu@sjtu.edu.cn
 # Please ask for permission before quote the code.
### 


# 询问是否进行批量测试，防止误操作
read -p "Are you sure to run the batch test? [yes/no]" answer
if [ $answer != "yes" ]; then
    echo "Exit the batch test."
    return
fi

## -------------------------- 性能监控 -------------------------- ##
# 初始化临时文件
cpu_file=$(mktemp)
mem_file=$(mktemp)
count_file=$(mktemp)

# 初始化变量
echo "0" > $cpu_file
echo "0" > $mem_file
echo "0" > $count_file

# 获取当前脚本的PID
pid=$$

# 定义采样函数
sample() {
  # 使用 ps 命令获取当前进程的CPU和内存使用情况
  # cpu=$(ps -p $pid -o %cpu=)
  # mem=$(ps -p $pid -o %mem=)
  # 使用 top 命令获取系统的CPU和内存使用情况
  # `top -bn1` 运行 top 命令一次，并输出到标准输出
  cpu=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')
  # mem=$(top -bn1 | grep "MiB Mem" | awk '{print $8/$4 * 100.0}')
  mem=$(top -bn1 | grep "MiB Mem" | awk '{print $8/1024}')
  # 使用 vmstat 命令获取系统的CPU和内存使用情况
  # `vmstat 1 2` 采样两次，忽略第一次的值
  # read cpu mem <<< $(vmstat 1 2 | tail -1 | awk '{print 100-$15, $3/$4 * 100}')

  # 累积计算总CPU和内存使用
  cpu_sum=$(cat $cpu_file)
  mem_sum=$(cat $mem_file)
  count=$(cat $count_file)

  cpu_sum=$(echo "$cpu_sum + $cpu" | bc)
  mem_sum=$(echo "$mem_sum + $mem" | bc)

  # 更新CPU和内存的最大值
  cpu_max=$(cat $cpu_file)
  mem_max=$(cat $mem_file)
  if (( $(echo "$cpu > $cpu_max" | bc -l) )); then
    cpu_max=$cpu
  fi
  if (( $(echo "$mem > $mem_max" | bc -l) )); then
    mem_max=$mem
  fi

  # 写回数据
  echo $cpu_sum > $cpu_file
  echo $mem_sum > $mem_file
  echo $cpu_max > $cpu_file
  echo $mem_max > $mem_file
  echo $((count + 1)) > $count_file
}

# 设置采样间隔（秒）
interval=1

# 在后台运行采样循环
while true; do
  sample
  sleep $interval
done &
mon_pid=$!
## -------------------------- 性能监控 -------------------------- ##

# 设定并行度限制
PARALLEL_LIMIT=32
# 存储后台作业PID的数组
background_pids=()
override_log_path="logs"
date_str=$(date +%F_%H-%M-%S)
err_log_path="err_logs_ablation_${date_str}"

# 如果err_log_path文件夹不存在，则新建
if [ ! -d ${err_log_path} ]; then
	mkdir ${err_log_path}
fi
if [ ! -d ${override_log_path} ]; then
	mkdir ${override_log_path}
fi

echo "Tasks begin at: $date_str" > $err_log_path/run.log

# 启动作业的函数
start_job() {
    {
        echo "启动进程: TW=$1, Dataset=$2, Dense Mode=$3, Dyn STW=$4, Speculative=$5, CSR Compress=$6"
        # 复制demo.cfg文件到/tmp/demo_$1_$2_$3_$4_$5_$6.cfg
		cp demo.cfg /tmp/demo_$1_$2_$3_$4_$5_$6.cfg
		# 修改/tmp/demo_$1_$2_$3_$4_$5_$6.cfg文件中的parallel_granularity参数
		sed -i "s/dense_mode = \".*\"/dense_mode = \"$3\"/g" /tmp/demo_$1_$2_$3_$4_$5_$6.cfg
		sed -i "s/dynamic_stw = .*/dynamic_stw = $4/g" /tmp/demo_$1_$2_$3_$4_$5_$6.cfg
		sed -i "s/speculative = .*/speculative = $5/g" /tmp/demo_$1_$2_$3_$4_$5_$6.cfg
		sed -i "s/csr_compress = .*/csr_compress = $6/g" /tmp/demo_$1_$2_$3_$4_$5_$6.cfg
		# 修改/tmp/demo_$1_$2_$3_$4_$5_$6.cfg文件中的TW参数
		if [ $2 == "DVS128Gesture" ]; then
			sed -i "s/mergedTimestep = .*/mergedTimestep = 100/g" /tmp/demo_$1_$2_$3_$4_$5_$6.cfg
		elif [ $2 == "CIFAR10DVS" ]; then
			sed -i "s/mergedTimestep = .*/mergedTimestep = 100/g" /tmp/demo_$1_$2_$3_$4_$5_$6.cfg
		elif [ $2 == "IMAGENET" ]; then
			sed -i "s/mergedTimestep = .*/mergedTimestep = 100/g" /tmp/demo_$1_$2_$3_$4_$5_$6.cfg
		fi
		sed -i "s/sub_tw = .*/sub_tw = $1/g" /tmp/demo_$1_$2_$3_$4_$5_$6.cfg
		# 判断layer_record_$2/trace_command_with_cfg_$1_$3_$4_$5_$6.sh是否存在，如果存在则删除，并从layer_record_$2/trace_command.sh复制
		if [ -f "layer_record_$2/trace_command_with_cfg_$1_$3_$4_$5_$6.sh" ]; then
			/usr/bin/rm layer_record_$2/trace_command_with_cfg_$1_$3_$4_$5_$6.sh
		fi
		cp layer_record_$2/trace_command.sh layer_record_$2/trace_command_with_cfg_$1_$3_$4_$5_$6.sh
		# "tr -d"和echo中"-n"避免后续追加换行
		head -n 1 layer_record_$2/trace_command.sh | tr -d '\n' > layer_record_$2/trace_command_with_cfg_$1_$3_$4_$5_$6.sh
		# 在layer_record_$2/trace_command_with_cfg_$1_$3_$4.sh文件末尾添加" --config /tmp/demo_$1_$2_$3_$4_$5_$6.cfg"
		echo -n " --config /tmp/demo_$1_$2_$3_$4_$5_$6.cfg" >> layer_record_$2/trace_command_with_cfg_$1_$3_$4_$5_$6.sh
		echo -n " --log_path $override_log_path" >> layer_record_$2/trace_command_with_cfg_$1_$3_$4_$5_$6.sh
		echo " 2> $err_log_path/PyNeuroSim_$1_$2_$3_$4_$5_$6.err" >> layer_record_$2/trace_command_with_cfg_$1_$3_$4_$5_$6.sh
		echo "source layer_record_$2/trace_command_with_cfg_$1_$3_$4_$5_$6.sh"
		source layer_record_$2/trace_command_with_cfg_$1_$3_$4_$5_$6.sh
		/usr/bin/rm layer_record_$2/trace_command_with_cfg_$1_$3_$4_$5_$6.sh

		# Test: random sleep for 5 to 10 seconds
		# sleep $[ ( $RANDOM % 6 )  + 5 ]s
    } &
    # 获取最新后台作业的PID并存储到数组
    pid=$!
    background_pids+=($pid)
}

# 检查并更新后台作业状态的函数
check_jobs() {
    for pid in "${background_pids[@]}"; do
        if ! kill -0 $pid 2>/dev/null; then
            # 如果作业已经完成，从数组中移除对应的PID
            background_pids=(${background_pids[@]/$pid})
        fi
    done
}


# dense_modes=("SpinalFlow" "PTB" "Strawman" "ComPASS")
SubTWs=(0 1 8 16 24 32 48 64 100)
datasets=("DVS128GestureSimplify" "CIFAR10DVS" "IMAGENET")
dense_modes=("PTB" "ComPASS" "Strawman")
dyn_stw=("True" "False")

speculative=("True" "False")
csr_compress=("True" "False")

count=0
for tw in ${SubTWs[*]}
do
	for dataset in ${datasets[*]}
	do
		for dm in ${dense_modes[*]}
		do
			for dstw in ${dyn_stw[*]}
			do
				for spec in ${speculative[*]}
				do
					for csr in ${csr_compress[*]}
					do
						# 仅在合理的参数组合下启动后台作业
						if [ $tw -eq 0 ] && ( [ $dm != "Strawman" ] && [ $dstw != "True" ] ); then
							continue
						fi
						if [ $dstw == "True" ] && ( [ $dm != "ComPASS" ] || [ $tw -ne 0 ] ); then
							continue
						fi
						if [ $tw -gt 0 ] && ( [ $dstw == "True" ] || [ $dm == "Strawman" ] ); then
							continue
						fi
						if [ $spec == "True" ] && ( [ $dm != "ComPASS" ] || [ $tw -ne 0 ] ); then
							continue
						fi
						if [ $csr == "True" ] && ( [ $dm != "ComPASS" ] || [ $tw -ne 0 ] ); then
							continue
						fi

						count=$((count+1))
						
						# 启动后台作业
						echo "启动后台作业: TW=$tw, Dataset=$dataset, Dense Mode=$dm, Dyn STW=$dstw, Speculative=$spec, CSR Compress=$csr" 
						start_job $tw $dataset $dm $dstw $spec $csr

						# 检查并更新后台作业状态
						check_jobs

						# 如果达到并行度限制，则等待至少一个后台作业完成
						while [ "${#background_pids[@]}" -ge "$PARALLEL_LIMIT" ]; do
							sleep 10 # 简单等待
							check_jobs # 再次检查并更新作业状态
						done
					done
				done
			done
		done
	done
done
echo "Total $count jobs are launched."

# 等待所有后台作业完成
for pid in "${background_pids[@]}"; do
    wait $pid
done

date_str=$(date +%F_%H-%M-%S)
echo "Tasks end at: $date_str" >> $err_log_path/run.log

## -------------------------- 性能监控 -------------------------- ##
# 杀掉采样循环进程
kill $mon_pid

# 从文件中读取数据
cpu_sum=$(cat $cpu_file)
mem_sum=$(cat $mem_file)
cpu_max=$(cat $cpu_file)
mem_max=$(cat $mem_file)
count=$(cat $count_file)

# 计算平均值并输出结果
if [ $count -ne 0 ]; then
  cpu_avg=$(echo "scale=2; $cpu_sum / $count" | bc)
  mem_avg=$(echo "scale=2; $mem_sum / $count" | bc)

  echo "CPU 平均使用率: $cpu_avg %" >> $err_log_path/run.log
  echo "CPU 峰值使用率: $cpu_max %" >> $err_log_path/run.log
  echo "内存平均使用量: $mem_avg GB" >> $err_log_path/run.log
  echo "内存峰值使用量: $mem_max GB" >> $err_log_path/run.log
else
  echo "没有采集到任何数据。" >> $err_log_path/run.log
fi

# 清理临时文件
rm $cpu_file $mem_file $count_file
## -------------------------- 性能监控 -------------------------- ##
