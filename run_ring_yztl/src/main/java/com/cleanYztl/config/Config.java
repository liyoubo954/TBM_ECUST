package com.cleanYztl.config;

import lombok.Data;
import org.apache.flink.api.java.utils.ParameterTool;

import com.cleanYztl.constant.ParameterKeyConst;

import java.io.Serializable;
import java.util.Objects;

@Data
public class Config implements Serializable {
    private static final long serialVersionUID = 1L;
    
    private final String jobName;
    private final InfluxDBConfig influxDBConfig;

    private Config(String jobName, InfluxDBConfig influxDBConfig) {
        this.jobName = jobName;
        this.influxDBConfig = influxDBConfig;
    }

    public static Config parse(ParameterTool params) {
        Objects.requireNonNull(params, "ParameterTool cannot be null");
        return new Config(
            params.get(ParameterKeyConst.SERVER_NAME, "ring-average-processor"), 
            InfluxDBConfig.fromParams(params)
        );
    }
}
