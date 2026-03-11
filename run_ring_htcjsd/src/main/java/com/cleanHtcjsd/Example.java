package com.cleanHtcjsd;

import org.apache.flink.api.java.utils.ParameterTool;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.cleanHtcjsd.config.Config;
import com.cleanHtcjsd.config.ExecutionEnvironmentConfig;
import com.cleanHtcjsd.source.InfluxDBRingSource;

public class Example {
    private static final Logger LOG = LoggerFactory.getLogger(Example.class);

    public static void main(String[] args) throws Exception {
        System.err.println("[main] main called");
        LOG.error("[main] main called");
        ParameterTool params = ParameterTool.fromArgs(args);
        Config config = Config.parse(params);
        String jobName = config.getJobName();
        StreamExecutionEnvironment env = ExecutionEnvironmentConfig.createEnv(params);
        System.err.println("[main] before InfluxDBRingSource.start");
        LOG.error("[main] before InfluxDBRingSource.start");
        new InfluxDBRingSource(config.getInfluxDBConfig()).start(env, "influxdbRingSource");
        System.err.println("[main] after InfluxDBRingSource.start");
        LOG.error("[main] after InfluxDBRingSource.start");
        // 注册关闭 Hook
        addShutdownHook();
        // 启动 Flink 作业
        LOG.info("Job '{}' execute...", jobName);
        System.err.println("[main] before env.executeAsync");
        env.executeAsync(jobName);
        System.err.println("[main] after env.executeAsync");
    }

    public static void addShutdownHook() {
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            LOG.info("Shutdown hook triggered. Cleaning up resources...");
            System.err.println("[main] Shutdown hook triggered. Cleaning up resources...");
        }));
    }
}
