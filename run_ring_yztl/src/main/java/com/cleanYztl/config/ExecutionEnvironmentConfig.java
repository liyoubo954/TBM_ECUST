package com.cleanYztl.config;

import static com.cleanYztl.constant.ParameterKeyConst.EXECUTION_MODE;

import org.apache.flink.api.common.RuntimeExecutionMode;
import org.apache.flink.api.java.utils.ParameterTool;
import org.apache.flink.core.execution.CheckpointingMode;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

public class ExecutionEnvironmentConfig {

    public static StreamExecutionEnvironment createEnv(ParameterTool params) {//ParameterTool 参数工具类，用来接受参数
        final StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        RuntimeExecutionMode mode = RuntimeExecutionMode.STREAMING;//默认使用流处理模式
        if (params.has(EXECUTION_MODE)) {
            mode = RuntimeExecutionMode.valueOf(params.get(EXECUTION_MODE).toUpperCase());
        }
        env.setRuntimeMode(mode);
        // CheckpointingMode.EXACTLY_ONCE: 开销较大，每条数据只会被处理一次，无重复也无遗漏
        // CheckpointingMode.AT_LEAST_ONCE: 每条数据至少被处理一次，可能会有重复，但不会丢失数据
        env.enableCheckpointing(5000, CheckpointingMode.AT_LEAST_ONCE);
        env.getCheckpointConfig().setCheckpointTimeout(60000);
        env.getCheckpointConfig().setMaxConcurrentCheckpoints(1);//checkpoint配置，是为了保障恢复时数据的一致性
        return env;
    }
}
