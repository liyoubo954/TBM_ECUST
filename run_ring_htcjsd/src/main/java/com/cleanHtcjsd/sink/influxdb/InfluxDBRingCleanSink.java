package com.cleanHtcjsd.sink.influxdb;

import org.apache.flink.api.connector.sink2.Sink;
import org.apache.flink.api.connector.sink2.SinkWriter;
import org.influxdb.InfluxDB;
import org.influxdb.InfluxDBFactory;
import org.influxdb.dto.Point;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.cleanHtcjsd.config.InfluxDBConfig;
import com.cleanHtcjsd.models.RingCleanData;

import okhttp3.OkHttpClient;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

public class InfluxDBRingCleanSink implements Sink<RingCleanData> {
    private final InfluxDBConfig config;

    public InfluxDBRingCleanSink(InfluxDBConfig config) {
        this.config = config;
    }

    @Override
    public SinkWriter<RingCleanData> createWriter(Sink.InitContext context) throws IOException {
        return new InfluxDBRingCleanWriter(config);
    }

    public static class InfluxDBRingCleanWriter implements SinkWriter<RingCleanData> {
        private static final Logger LOG = LoggerFactory.getLogger(InfluxDBRingCleanWriter.class);
        private static final int MAX_RETRIES = 3;
        private static final long RETRY_DELAY_MS = 1000;

        private final List<RingCleanData> batchBuffer = new ArrayList<>();
        private final InfluxDB influxDB;
        private final int batchSize;
        private final String targetDatabase = "htcjsd_dz1368_cleanRing";

        public InfluxDBRingCleanWriter(InfluxDBConfig config) {
            this.batchSize = config.getBatchSize();
            
            // 连接到InfluxDB
            OkHttpClient.Builder httpBuilder = new OkHttpClient.Builder()
                .connectTimeout(config.getConnectionTimeout(), TimeUnit.MILLISECONDS)
                .readTimeout(config.getConnectionTimeout(), TimeUnit.MILLISECONDS)
                .writeTimeout(config.getConnectionTimeout(), TimeUnit.MILLISECONDS);
            this.influxDB = InfluxDBFactory.connect(
                    config.getUrl(),
                    config.getUser(),
                    config.getPassword(),
                    httpBuilder
            );
            
            // 确保目标数据库存在（若不存在则创建）
            try {
                influxDB.createDatabase(targetDatabase);
            } catch (Exception e) {
                LOG.info("Target database '{}' may already exist or cannot be created: {}", targetDatabase, e.getMessage());
            }
            
            // 设置目标数据库
            this.influxDB.setDatabase(targetDatabase);
            this.influxDB.enableBatch(
                config.getBatchSize(),
                config.getConnectionTimeout().intValue(),
                TimeUnit.MILLISECONDS
            );
            
            LOG.info("InfluxDBRingCleanWriter initialized - url: {}, database: {}", 
                    config.getUrl(), targetDatabase);
        }

        private void flushBatch() throws IOException {
            if (batchBuffer.isEmpty()) {
                return;
            }
            
            int retryCount = 0;
            while (retryCount < MAX_RETRIES) {
                try {
                    for (RingCleanData data : batchBuffer) {
                        // 构建measurement名称：Ring_ + 环号
                        String measurementName = "Ring_" + data.getRing();
                        
                        // 解析时间戳（纳秒）
                        long timestampNs = parseTimestamp(data.getTime());
                        
                        // 构建Point，动态添加所有字段
                        Point.Builder pointBuilder = Point.measurement(measurementName);
                        
                        // 动态添加所有字段，保持原始字段名和数据类型不变
                        Map<String, Object> fields = data.getFields();
                        for (Map.Entry<String, Object> entry : fields.entrySet()) {
                            String fieldName = entry.getKey();
                            Object value = entry.getValue();
                            
                            // 跳过null值（已经经过清洗，不应该有null值）
                            if (value == null) {
                                continue;
                            }
                            
                            // 根据值的类型添加字段，保持原始类型
                            if (value instanceof Number) {
                                // 数字类型直接添加
                                pointBuilder.addField(fieldName, (Number) value);
                            } else if (value instanceof Boolean) {
                                // 布尔类型
                                pointBuilder.addField(fieldName, (Boolean) value);
                            } else if (value instanceof String) {
                                // 字符串类型，跳过空字符串
                                String strValue = (String) value;
                                if (!strValue.trim().isEmpty()) {
                                    pointBuilder.addField(fieldName, strValue);
                                }
                            } else {
                                // 其他类型转为字符串
                                pointBuilder.addField(fieldName, value.toString());
                            }
                        }
                        
                        // 设置时间戳（纳秒）
                        Point point = pointBuilder.time(timestampNs, TimeUnit.NANOSECONDS).build();
                        
                        influxDB.write(point);
                    }
                    
                    LOG.info("Successfully flushed batch of size: {} to database: {}", batchBuffer.size(), targetDatabase);
                    batchBuffer.clear();
                    return;
                } catch (Exception e) {
                    retryCount++;
                    if (retryCount < MAX_RETRIES) {
                        LOG.warn("Retry {} of {} after error: {}", retryCount, MAX_RETRIES, e.getMessage());
                        try {
                            Thread.sleep(RETRY_DELAY_MS);
                        } catch (InterruptedException ie) {
                            Thread.currentThread().interrupt();
                            throw new IOException("Interrupted during retry", ie);
                        }
                    } else {
                        throw new IOException("Failed to write batch after " + MAX_RETRIES + " retries", e);
                    }
                }
            }
        }

        /**
         * 解析时间戳字符串为纳秒数
         */
        private long parseTimestamp(String timeStr) {
            if (timeStr == null || timeStr.trim().isEmpty()) {
                // 如果没有时间戳，使用当前时间（epoch 纳秒）
                return System.currentTimeMillis() * 1_000_000L;
            }
            
            try {
                // 纯数字：判断单位并统一转换为纳秒
                if (timeStr.matches("\\d+")) {
                    long v = Long.parseLong(timeStr);
                    // 10位左右：秒 -> ns
                    if (v > 0 && v < 10_000_000_000L) {
                        return v * 1_000_000_000L;
                    }
                    // 13位左右：毫秒 -> ns
                    if (v >= 10_000_000_000L && v < 10_000_000_000_000L) {
                        return v * 1_000_000L;
                    }
                    // 16位左右：微秒 -> ns
                    if (v >= 10_000_000_000_000L && v < 10_000_000_000_000_000L) {
                        return v * 1_000L;
                    }
                    // 19位左右：纳秒
                    return v;
                }
                // ISO-8601/RFC3339：例如 2025-07-17T07:29:05Z 或带纳秒小数
                java.time.Instant instant = java.time.Instant.parse(timeStr);
                return instant.getEpochSecond() * 1_000_000_000L + instant.getNano();
            } catch (NumberFormatException e) {
                LOG.warn("Could not parse numeric timestamp: {}, using current epoch time", timeStr, e);
                return System.currentTimeMillis() * 1_000_000L;
            } catch (java.time.format.DateTimeParseException e) {
                LOG.warn("Could not parse RFC3339 timestamp: {}, using current epoch time", timeStr, e);
                return System.currentTimeMillis() * 1_000_000L;
            }
        }

        @Override
        public void close() throws Exception {
            try {
                if (!batchBuffer.isEmpty()) {
                    flushBatch();
                }
            } finally {
                if (influxDB != null) {
                    influxDB.close();
                }
            }
        }

        @Override
        public void write(RingCleanData data, Context context) throws IOException, InterruptedException {
            if (data == null) return;
            
            batchBuffer.add(data);
            if (batchBuffer.size() >= batchSize) {
                flushBatch();
            }
        }

        @Override
        public void flush(boolean endOfInput) throws IOException, InterruptedException {
            flushBatch();
        }
    }
}
