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

# 设定并行度限制
PARALLEL_LIMIT=8
# 存储后台作业PID的数组
background_pids=()
override_log_path="logs"
date_str=$(date +%F_%H-%M-%S)
err_log_path="err_logs_${date_str}"

# 询问是否进行批量测试，防止误操作
read -p "Are you sure to run the batch test? [yes/no]" answer
if [ $answer != "yes" ]; then
    echo "Exit the batch test."
    return
fi

# 启动作业的函数
start_job() {
    {
        echo "启动进程: TW=$1, Dataset=$2, Dense Mode=$3, Dyn STW=$4"
        # 复制demo.cfg文件到/tmp/demo_$1_$2_$3_$4.cfg
		cp demo.cfg /tmp/demo_$1_$2_$3_$4.cfg
		# 修改/tmp/demo_$1_$2_$3_$4.cfg文件中的parallel_granularity参数
		sed -i "s/dense_mode = \".*\"/dense_mode = \"$3\"/g" /tmp/demo_$1_$2_$3_$4.cfg
		sed -i "s/dynamic_stw = .*/dynamic_stw = $4/g" /tmp/demo_$1_$2_$3_$4.cfg
		# 修改/tmp/demo_$1_$2_$3_$4.cfg文件中的TW参数
		if [ $2 == "DVS128Gesture" ]; then
			sed -i "s/mergedTimestep = .*/mergedTimestep = 100/g" /tmp/demo_$1_$2_$3_$4.cfg
		elif [ $2 == "CIFAR10DVS" ]; then
			sed -i "s/mergedTimestep = .*/mergedTimestep = 100/g" /tmp/demo_$1_$2_$3_$4.cfg
		elif [ $2 == "IMAGENET" ]; then
			sed -i "s/mergedTimestep = .*/mergedTimestep = 100/g" /tmp/demo_$1_$2_$3_$4.cfg
		elif [ $2 == "IMAGENET_RN50" ]; then
			sed -i "s/mergedTimestep = .*/mergedTimestep = 100/g" /tmp/demo_$1_$2_$3_$4.cfg
		fi
		sed -i "s/sub_tw = .*/sub_tw = $1/g" /tmp/demo_$1_$2_$3_$4.cfg
		# 判断layer_record_$2/trace_command_with_cfg_$1_$3_$4.sh是否存在，如果存在则删除，并从layer_record_$2/trace_command.sh复制
		if [ -f "layer_record_$2/trace_command_with_cfg_$1_$3_$4.sh" ]; then
			/usr/bin/rm layer_record_$2/trace_command_with_cfg_$1_$3_$4.sh
		fi
		cp layer_record_$2/trace_command.sh layer_record_$2/trace_command_with_cfg_$1_$3_$4.sh
		# "tr -d"和echo中"-n"避免后续追加换行
		head -n 1 layer_record_$2/trace_command.sh | tr -d '\n' > layer_record_$2/trace_command_with_cfg_$1_$3_$4.sh
		# 在layer_record_$2/trace_command_with_cfg_$1_$3_$4.sh文件末尾添加" --config /tmp/demo_$1_$2_$3_$4.cfg"
		echo -n " --config /tmp/demo_$1_$2_$3_$4.cfg" >> layer_record_$2/trace_command_with_cfg_$1_$3_$4.sh
		echo -n " --log_path $override_log_path" >> layer_record_$2/trace_command_with_cfg_$1_$3_$4.sh
		echo " 2> $err_log_path/PyNeuroSim_$1_$2_$3_$4.err" >> layer_record_$2/trace_command_with_cfg_$1_$3_$4.sh
		echo "source layer_record_$2/trace_command_with_cfg_$1_$3_$4.sh"
		# source layer_record_$2/trace_command_with_cfg_$1_$3_$4.sh
		# /usr/bin/rm layer_record_$2/trace_command_with_cfg_$1_$3_$4.sh

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

# 如果err_log_path文件夹不存在，则新建
if [ ! -d ${err_log_path} ]; then
	mkdir ${err_log_path}
fi


# dense_modes=("SpinalFlow" "PTB" "Strawman" "ComPASS")
# SubTWs=(0 1 8 16 24 32 48 64 100)
SubTWs=(8)
# datasets=("DVS128GestureSimplify" "CIFAR10DVS" "IMAGENET")
datasets=("IMAGENET")
# dense_modes=("PTB" "Strawman" "ComPASS")
dense_modes=("PTB")
dyn_stw=("False")

count=0
for tw in ${SubTWs[*]}
do
	for dataset in ${datasets[*]}
	do
		for dm in ${dense_modes[*]}
		do
			for dstw in ${dyn_stw[*]}
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

				count=$((count+1))
				
				# 启动后台作业
				echo "启动后台作业: TW=$tw, Dataset=$dataset, Dense Mode=$dm, Dyn STW=$dstw"
				start_job $tw $dataset $dm $dstw

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
echo "Total $count jobs are launched."

# 等待所有后台作业完成
for pid in "${background_pids[@]}"; do
    wait $pid
done
