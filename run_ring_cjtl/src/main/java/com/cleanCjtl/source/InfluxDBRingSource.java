package com.cleanCjtl.source;

import org.influxdb.InfluxDB;
import org.influxdb.InfluxDBFactory;
import org.influxdb.dto.Query;
import org.influxdb.dto.QueryResult;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.source.RichSourceFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.cleanCjtl.config.InfluxDBConfig;
import com.cleanCjtl.models.RingCleanData;

import okhttp3.OkHttpClient;

import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

public class InfluxDBRingSource extends RichSourceFunction<RingCleanData> {
    private static final Logger LOG = LoggerFactory.getLogger(InfluxDBRingSource.class);
    private static final long serialVersionUID = 1L;

    private final InfluxDBConfig influxDBConfig;
    private final String sourceDatabase = "cjtl_dg1213_ring";
    private transient InfluxDB influxDB;
    private volatile boolean running = true;
    
    // 记录已处理的历史环号
    private Set<Integer> processedHistoryRings = new HashSet<>();
    // 记录每环的最后处理时间戳（纳秒）
    private Map<Integer, Long> ringLastTimestampMap = new HashMap<>();
    // 当前处理阶段：历史数据处理 或 实时数据处理
    private boolean isProcessingHistory = true;
    // 最新一环的环号
    private Integer latestRingNumber = null;

    public InfluxDBRingSource(InfluxDBConfig influxDBConfig) {
        this.influxDBConfig = influxDBConfig;
    }

    @Override
    public void open(Configuration parameters) throws Exception {
        System.err.println("[InfluxDBRingSource] open called");
        LOG.error("[InfluxDBRingSource] open called");
        LOG.info("InfluxDBRingSource open() - url: {}, user: {}, database: {}", 
                influxDBConfig.getUrl(), influxDBConfig.getUser(), sourceDatabase);
        
        OkHttpClient.Builder httpBuilder = new OkHttpClient.Builder()
            .connectTimeout(influxDBConfig.getConnectionTimeout(), TimeUnit.MILLISECONDS)
            .readTimeout(influxDBConfig.getConnectionTimeout(), TimeUnit.MILLISECONDS)
            .writeTimeout(influxDBConfig.getConnectionTimeout(), TimeUnit.MILLISECONDS);
        influxDB = InfluxDBFactory.connect(
                influxDBConfig.getUrl(),
                influxDBConfig.getUser(),
                influxDBConfig.getPassword(),
                httpBuilder
        );
        influxDB.setDatabase(sourceDatabase);
        influxDB.enableBatch(influxDBConfig.getBatchSize(), 
                           influxDBConfig.getConnectionTimeout().intValue(), 
                           TimeUnit.MILLISECONDS);
        LOG.info("InfluxDBRingSource open() - Connected to InfluxDB and set database: {}", sourceDatabase);
    }

    @Override
    public void run(SourceContext<RingCleanData> ctx) throws Exception {
        System.err.println("[InfluxDBRingSource] run called");
        LOG.error("[InfluxDBRingSource] run called");
        
        while (running) {
            try {
                // 1. 查询所有measurement
                String showMeasurementsQuery = "SHOW MEASUREMENTS";
                QueryResult measurementsResult = influxDB.query(new Query(showMeasurementsQuery, sourceDatabase));
                
                // 2. 获取所有环号
                List<Integer> allRingNumbers = getAllRingNumbers(measurementsResult);
                if (allRingNumbers.isEmpty()) {
                    LOG.warn("No Ring measurements found in database: {}", sourceDatabase);
                    TimeUnit.SECONDS.sleep(30);
                    continue;
                }
                
                // 排序环号
                Collections.sort(allRingNumbers);
                
                // 3. 确定最新一环
                Integer maxRingNumber = Collections.max(allRingNumbers);
                
                // 4. 处理逻辑
                if (isProcessingHistory) {
                    // 处理历史数据：从最小环号开始，一次只处理一个环号
                    boolean hasHistoryData = false;
                    Integer nextRingToProcess = null;
                    
                    // 找到最小未处理的环号（排除最新一环）
                    for (Integer ringNumber : allRingNumbers) {
                        if (ringNumber.equals(maxRingNumber)) {
                            // 跳过最新一环，留待实时处理
                            continue;
                        }
                        
                        if (!processedHistoryRings.contains(ringNumber)) {
                            // 找到第一个未处理的环号（因为已排序，这就是最小的）
                            nextRingToProcess = ringNumber;
                            hasHistoryData = true;
                            break; // 只处理一个环号，严格按照顺序
                        }
                    }
                    
                    // 如果找到未处理的环号，处理它
                    if (nextRingToProcess != null) {
                        LOG.info("Processing history data for ring: {} (in order from smallest to largest)", nextRingToProcess);
                        processHistoryRing(ctx, nextRingToProcess);
                        processedHistoryRings.add(nextRingToProcess);
                        LOG.info("Completed processing ring: {}. Will process next ring in next cycle.", nextRingToProcess);
                    }
                    
                    // 如果所有历史环都处理完了，切换到实时处理模式
                    if (!hasHistoryData && maxRingNumber != null) {
                        isProcessingHistory = false;
                        latestRingNumber = maxRingNumber;
                        LOG.info("History data processing completed. Switching to real-time processing for ring: {}", latestRingNumber);
                    }
                }
                
                // 5. 实时处理：处理最新一环，每30秒一次
                if (!isProcessingHistory && maxRingNumber != null) {
                    // 检查是否有新的一环出现
                    if (!maxRingNumber.equals(latestRingNumber)) {
                        // 新环出现，先处理旧环的剩余历史数据
                        if (latestRingNumber != null && !processedHistoryRings.contains(latestRingNumber)) {
                            LOG.info("New ring detected. Processing remaining history data for old ring: {}", latestRingNumber);
                            processHistoryRing(ctx, latestRingNumber);
                            processedHistoryRings.add(latestRingNumber);
                        }
                        // 更新最新一环
                        latestRingNumber = maxRingNumber;
                        LOG.info("New ring detected: {}", latestRingNumber);
                    }
                    
                    // 处理最新一环的实时数据
                    processRealtimeRing(ctx, latestRingNumber);
                    
                    // 等待30秒
                    TimeUnit.SECONDS.sleep(30);
                } else if (isProcessingHistory) {
                    // 历史处理模式下，短暂等待后继续
                    TimeUnit.SECONDS.sleep(1);
                } else {
                    // 没有最新环，等待
                    TimeUnit.SECONDS.sleep(30);
                }
                
            } catch (Exception e) {
                LOG.error("Error in InfluxDBRingSource run()", e);
                System.err.println("Error in InfluxDBRingSource run(): " + e.getMessage());
                e.printStackTrace();
                TimeUnit.SECONDS.sleep(5);
            }
        }
    }

    /**
     * 处理历史环数据：拉取所有state=1的数据，删除包含null值的行
     */
    private void processHistoryRing(SourceContext<RingCleanData> ctx, Integer ringNumber) throws Exception {
        String measurementName = "Ring_" + ringNumber;
        String targetDatabase = "cjtl_dg1213_cleanRing";
        
        // 检查目标数据库是否已有该measurement的数据，避免重复处理
        try {
            String checkQuery = String.format(
                "SELECT COUNT(*) FROM %s",
                measurementName
            );
            QueryResult checkResult = influxDB.query(new Query(checkQuery, targetDatabase));
            
            if (checkResult != null && checkResult.getResults() != null && !checkResult.getResults().isEmpty()) {
                // 检查是否有数据
                for (QueryResult.Result result : checkResult.getResults()) {
                    if (result.getSeries() != null && !result.getSeries().isEmpty()) {
                        for (QueryResult.Series series : result.getSeries()) {
                            if (series.getValues() != null && !series.getValues().isEmpty()) {
                                for (List<Object> values : series.getValues()) {
                                    if (values != null && !values.isEmpty()) {
                                        Object countValue = values.get(0);
                                        if (countValue != null) {
                                            long count = 0;
                                            if (countValue instanceof Number) {
                                                count = ((Number) countValue).longValue();
                                            } else if (countValue instanceof String) {
                                                try {
                                                    count = Long.parseLong((String) countValue);
                                                } catch (NumberFormatException e) {
                                                    // 忽略解析错误
                                                }
                                            }
                                            if (count > 0) {
                                                LOG.info("Measurement {} already has {} records in target database. Skipping to avoid duplicate data.", 
                                                    measurementName, count);
                                                // 如果目标数据库已有数据，跳过处理，避免重复和覆盖
                                                return;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } catch (Exception e) {
            // 如果检查失败（可能是measurement不存在），继续处理
            LOG.debug("Failed to check existing data in target database for measurement: {}. Will proceed with processing. Error: {}", 
                measurementName, e.getMessage());
        }
        
        // 使用SELECT *查询所有字段，只查询state=1的数据
        String query = String.format(
            "SELECT * FROM %s WHERE state = 1",
            measurementName
        );
        
        LOG.info("Querying history data for measurement: {} with query: {}", measurementName, query);
        QueryResult queryResult = influxDB.query(new Query(query, sourceDatabase));
        
        if (queryResult.getResults() == null || queryResult.getResults().isEmpty()) {
            LOG.warn("No results returned from query for measurement: {}", measurementName);
            return;
        }
        
        // 解析数据
        List<RingCleanData> allData = parseQueryResult(queryResult, ringNumber);
        LOG.info("Parsed {} records from measurement: {}", allData.size(), measurementName);
        
        // 过滤：删除包含null值的行（检查所有字段）
        List<RingCleanData> cleanedData = filterNullValues(allData);
        LOG.info("After filtering null values, {} records remain for measurement: {}", cleanedData.size(), measurementName);
        
        // 记录每环的最大时间戳
        if (!cleanedData.isEmpty()) {
            Long maxTimestamp = cleanedData.stream()
                .map(data -> parseTimestamp(data.getTime()))
                .filter(Objects::nonNull)
                .max(Long::compareTo)
                .orElse(null);
            if (maxTimestamp != null) {
                ringLastTimestampMap.put(ringNumber, maxTimestamp);
                LOG.info("Recorded last timestamp for ring {}: {}", ringNumber, maxTimestamp);
            }
        }
        
        // 发送清洗后的数据
        for (RingCleanData data : cleanedData) {
            ctx.collect(data);
        }
        
        LOG.info("Processed {} cleaned records for history ring: {}", cleanedData.size(), ringNumber);
    }

    /**
     * 处理实时环数据：每30秒拉取一次，从上次时间戳的下一行开始
     */
    private void processRealtimeRing(SourceContext<RingCleanData> ctx, Integer ringNumber) throws Exception {
        String measurementName = "Ring_" + ringNumber;
        
        // 构建查询：state=1，且时间戳大于上次处理的时间戳
        String query;
        Long lastTimestamp = ringLastTimestampMap.get(ringNumber);
        
        if (lastTimestamp != null) {
            // 从上次时间戳的下一行开始，使用SELECT *查询所有字段
            query = String.format(
                "SELECT * FROM %s WHERE state = 1 AND time > %dns",
                measurementName,
                lastTimestamp
            );
        } else {
            // 第一次处理该环，查询所有state=1的数据
            query = String.format(
                "SELECT * FROM %s WHERE state = 1",
                measurementName
            );
        }
        
        LOG.info("Querying real-time data for measurement: {} with query: {}", measurementName, query);
        QueryResult queryResult = influxDB.query(new Query(query, sourceDatabase));
        
        if (queryResult.getResults() == null || queryResult.getResults().isEmpty()) {
            LOG.debug("No new data returned from query for measurement: {}", measurementName);
            return;
        }
        
        // 解析数据
        List<RingCleanData> allData = parseQueryResult(queryResult, ringNumber);
        LOG.info("Parsed {} new records from measurement: {}", allData.size(), measurementName);
        
        if (allData.isEmpty()) {
            return;
        }
        
        // 过滤：删除包含null值的行（检查所有字段）
        List<RingCleanData> cleanedData = filterNullValues(allData);
        LOG.info("After filtering null values, {} records remain for measurement: {}", cleanedData.size(), measurementName);
        
        // 更新最后处理的时间戳
        Long maxTimestamp = cleanedData.stream()
            .map(data -> parseTimestamp(data.getTime()))
            .filter(Objects::nonNull)
            .max(Long::compareTo)
            .orElse(lastTimestamp);
        
        if (maxTimestamp != null && (lastTimestamp == null || maxTimestamp > lastTimestamp)) {
            ringLastTimestampMap.put(ringNumber, maxTimestamp);
            LOG.info("Updated last timestamp for ring {}: {}", ringNumber, maxTimestamp);
        }
        
        // 发送清洗后的数据
        for (RingCleanData data : cleanedData) {
            ctx.collect(data);
        }
        
        LOG.info("Processed {} cleaned records for real-time ring: {}", cleanedData.size(), ringNumber);
    }

    /**
     * 过滤包含null值的行：检查行中所有字段，如果任何一个字段为null，则删除该行
     */
    private List<RingCleanData> filterNullValues(List<RingCleanData> dataList) {
        return dataList.stream()
            .filter(data -> {
                // 检查所有字段是否都不为null
                // 首先检查time字段
                if (data.getTime() == null || data.getTime().trim().isEmpty()) {
                    return false;
                }
                
                // 检查fields中的所有值是否都不为null
                for (Object value : data.getFields().values()) {
                    if (value == null) {
                        return false;
                    }
                    // 如果是字符串，检查是否为空
                    if (value instanceof String && ((String) value).trim().isEmpty()) {
                        return false;
                    }
                }
                
                return true;
            })
            .collect(Collectors.toList());
    }

    /**
     * 解析时间戳字符串为纳秒数
     * InfluxDB返回的时间戳通常是纳秒数（字符串格式）
     */
    private Long parseTimestamp(String timeStr) {
        if (timeStr == null || timeStr.trim().isEmpty()) {
            return null;
        }
        
        try {
            // 数字：根据位数判断单位并转换为纳秒
            if (timeStr.matches("\\d+")) {
                long v = Long.parseLong(timeStr);
                if (v > 0 && v < 10_000_000_000L) {
                    return v * 1_000_000_000L; // 秒 -> 纳秒
                }
                if (v >= 10_000_000_000L && v < 10_000_000_000_000L) {
                    return v * 1_000_000L; // 毫秒 -> 纳秒
                }
                if (v >= 10_000_000_000_000L && v < 10_000_000_000_000_000L) {
                    return v * 1_000L; // 微秒 -> 纳秒
                }
                return v; // 纳秒
            }
            // RFC3339/ISO-8601 字符串
            java.time.Instant instant = java.time.Instant.parse(timeStr);
            return instant.getEpochSecond() * 1_000_000_000L + instant.getNano();
        } catch (NumberFormatException e) {
            LOG.warn("Could not parse numeric timestamp: {}", timeStr, e);
            return null;
        } catch (java.time.format.DateTimeParseException e) {
            LOG.warn("Could not parse RFC3339 timestamp: {}", timeStr, e);
            return null;
        }
    }

    /**
     * 获取所有环号
     */
    private List<Integer> getAllRingNumbers(QueryResult result) {
        List<Integer> ringNumbers = new ArrayList<>();
        Pattern pattern = Pattern.compile("Ring_(\\d+)");
        
        for (QueryResult.Result queryResult : result.getResults()) {
            if (queryResult.getSeries() == null) continue;
            
            for (QueryResult.Series series : queryResult.getSeries()) {
                for (List<Object> values : series.getValues()) {
                    for (Object value : values) {
                        if (value != null) {
                            String measurementName = value.toString();
                            Matcher matcher = pattern.matcher(measurementName);
                            if (matcher.find()) {
                                try {
                                    int ringNumber = Integer.parseInt(matcher.group(1));
                                    ringNumbers.add(ringNumber);
                                } catch (NumberFormatException e) {
                                    LOG.warn("Could not parse ring number from measurement: {}", measurementName);
                                }
                            }
                        }
                    }
                }
            }
        }
        
        return ringNumbers;
    }

    /**
     * 解析查询结果，动态处理所有字段
     */
    private List<RingCleanData> parseQueryResult(QueryResult result, Integer ringNumber) {
        List<RingCleanData> dataList = new ArrayList<>();
        int totalRows = 0;
        
        for (QueryResult.Result queryResult : result.getResults()) {
            if (queryResult.getSeries() == null) continue;
            
            for (QueryResult.Series series : queryResult.getSeries()) {
                List<String> columns = series.getColumns();
                LOG.debug("Available columns: {}", columns);
                
                Map<String, Integer> columnIndexMap = new HashMap<>();
                for (int i = 0; i < columns.size(); i++) {
                    columnIndexMap.put(columns.get(i), i);
                }
                
                for (List<Object> values : series.getValues()) {
                    totalRows++;
                    RingCleanData data = new RingCleanData();
                    data.setRing(ringNumber);
                    
                    // 动态解析所有字段（除了time字段）
                    for (String column : columns) {
                        if ("time".equalsIgnoreCase(column)) {
                            // time字段单独处理
                            Integer timeIndex = columnIndexMap.get(column);
                            if (timeIndex != null && timeIndex < values.size() && values.get(timeIndex) != null) {
                                Object timeValue = values.get(timeIndex);
                                // InfluxDB的时间戳是纳秒数，但返回时可能已经转换为字符串或数字
                                if (timeValue instanceof Number) {
                                    long timestampNs = ((Number) timeValue).longValue();
                                    // 如果时间戳看起来像毫秒（小于某个阈值），转换为纳秒
                                    if (timestampNs > 0 && timestampNs < 1_000_000_000_000L) {
                                        timestampNs = timestampNs * 1_000_000L;
                                    }
                                    data.setTime(String.valueOf(timestampNs));
                                } else {
                                    data.setTime(timeValue.toString());
                                }
                            }
                        } else {
                            // 其他字段添加到fields中
                            Integer index = columnIndexMap.get(column);
                            if (index != null && index < values.size()) {
                                Object value = values.get(index);
                                // 保持原始值的类型
                                data.addField(column, value);
                            }
                        }
                    }
                    
                    dataList.add(data);
                }
            }
        }
        
        LOG.debug("parseQueryResult: totalRows={}, dataList.size={}", totalRows, dataList.size());
        return dataList;
    }

    @Override
    public void cancel() {
        running = false;
    }

    @Override
    public void close() throws Exception {
        if (influxDB != null) {
            influxDB.close();
        }
    }

    public void start(StreamExecutionEnvironment env, String sourceName) throws Exception {
        LOG.info("InfluxDBRingSource start() - Registering source: {}", sourceName);

        DataStream<RingCleanData> sourceStream = env
            .addSource(this)
            .name(sourceName);

        // 直接sink到InfluxDB
        sourceStream
            .sinkTo(new com.cleanCjtl.sink.influxdb.InfluxDBRingCleanSink(influxDBConfig))
            .name("InfluxDBRingCleanSink");
        LOG.info("InfluxDBRingSource start() - Data processing pipeline registered.");
    }
}
