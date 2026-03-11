package com.cleanCjtl.models;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

/**
 * 清洗后的环数据模型，支持动态字段
 * 存储所有字段名和值的映射关系
 */
public class RingCleanData implements Serializable {
    private static final long serialVersionUID = 1L;

    // 存储所有字段的键值对
    private Map<String, Object> fields = new HashMap<>();
    
    // 时间戳（纳秒，字符串格式）
    private String time;
    
    // 环号
    private Integer ring;

    public RingCleanData() {
    }

    public Map<String, Object> getFields() {
        return fields;
    }

    public void setFields(Map<String, Object> fields) {
        this.fields = fields;
    }

    public void addField(String fieldName, Object value) {
        if (fieldName != null && !fieldName.equalsIgnoreCase("time")) {
            fields.put(fieldName, value);
        }
    }

    public Object getField(String fieldName) {
        return fields.get(fieldName);
    }

    public String getTime() {
        return time;
    }

    public void setTime(String time) {
        this.time = time;
    }

    public Integer getRing() {
        return ring;
    }

    public void setRing(Integer ring) {
        this.ring = ring;
    }

    /**
     * 检查所有字段是否都不为null（除了time字段）
     * @return 如果所有字段都不为null返回true，否则返回false
     */
    public boolean hasNoNullFields() {
        for (Object value : fields.values()) {
            if (value == null) {
                return false;
            }
        }
        return time != null && !time.trim().isEmpty();
    }
}
