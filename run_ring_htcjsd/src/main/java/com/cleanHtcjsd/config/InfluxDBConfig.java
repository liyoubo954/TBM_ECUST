package com.cleanHtcjsd.config;

import org.apache.flink.api.java.utils.ParameterTool;
import java.io.Serializable;

public class InfluxDBConfig implements Serializable {
    private static final long serialVersionUID = 1L;

    private static final String INFLUXDB_URL = "influxdb.url";
    private static final String INFLUXDB_USER = "influxdb.user";
    private static final String INFLUXDB_PASSWORD = "influxdb.password";
    private static final String INFLUXDB_DATABASE = "influxdb.database";
    private static final String INFLUXDB_BATCH_SIZE = "influxdb.batch.size";
    private static final String INFLUXDB_MAX_POOL_SIZE = "influxdb.max.pool.size";
    private static final String INFLUXDB_CONNECTION_TIMEOUT = "influxdb.connection.timeout";

    private final String url;
    private final String user;
    private final String password;
    private final String database;
    private final Integer batchSize;
    private final Integer maxPoolSize;
    private final Long connectionTimeout;

    private InfluxDBConfig(ParameterTool params) {
        this.url = params.getRequired(INFLUXDB_URL);
        this.user = params.get(INFLUXDB_USER);
        this.password = params.get(INFLUXDB_PASSWORD);
        this.database = params.getRequired(INFLUXDB_DATABASE);
        this.batchSize = params.getInt(INFLUXDB_BATCH_SIZE, 100);
        this.maxPoolSize = params.getInt(INFLUXDB_MAX_POOL_SIZE, 10);
        this.connectionTimeout = params.getLong(INFLUXDB_CONNECTION_TIMEOUT, 30000L);
    }

    public String getUrl() {
        return url;
    }

    public String getUser() {
        return user;
    }

    public String getPassword() {
        return password;
    }

    public String getDatabase() {
        return database;
    }

    public Integer getBatchSize() {
        return batchSize;
    }

    public Integer getMaxPoolSize() {
        return maxPoolSize;
    }

    public Long getConnectionTimeout() {
        return connectionTimeout;
    }

    public static InfluxDBConfig fromParams(ParameterTool params) {
        return new InfluxDBConfig(params);
    }
} 