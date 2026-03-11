# flink examples

## 环境

- OS: Debian 12.10
- Docker: 28.0.1, build 068a01e
- Java: OpenJDK 17.0.14 2025-01-21
- Maven: Apache Maven 3.8.7
- Flink: 1.20.1
- 数据库：PostgreSQL 17.4
- 时序数据库：influxdb 1.11.8

## 主要内容

- 日志规范：禁止打印敏感内容（密码、密钥等），调试内容、提示信息、错误（异常）内容
- 版本管理：<https://semver.org/lang/zh-CN/>
- [Flink 基础概念](flink.md)

## 环境初始化

### 环境启动

```shell
export HOST_IP=$(hostname -I | awk '{print $1}')
# 查看宿主机IP是否正确，HOST_IP 影响访问kafka
echo ${HOST_IP}
docker compose up -d
```

### kafka

#### 进入kafka容器

```shell
docker exec -it kafka bash
```

#### 创建主题

```shell
# 容器内执行
export HOST_IP=$(hostname -I | awk '{print $1}')
kafka-topics.sh --create \
  --topic example-data-num \
  --bootstrap-server ${HOST_IP}:9092 \
  --partitions 1 \
  --replication-factor 1
```

#### 发送测试数据

```shell
# 容器内执行
export HOST_IP=$(hostname -I | awk '{print $1}')
kafka-console-producer.sh --topic example-data-num --bootstrap-server ${HOST_IP}:9092
```

#### 消费数据

```shell
# 容器内执行
export HOST_IP=$(hostname -I | awk '{print $1}')
kafka-console-consumer.sh --topic example-data-num --bootstrap-server ${HOST_IP}:9092 --from-beginning
```

### postgres

#### 进入postgres容器

```shell
docker exec -it postgres bash
# 容器内执行
psql -U postgres
```

#### 建表语句

```SQL
DROP TABLE IF EXISTS example_data_t;
CREATE TABLE IF NOT EXISTS example_data_t (
 id varchar(40) NOT NULL,
 "val" numeric NOT NULL,
 create_at timestamptz DEFAULT now() NULL,
 process_at timestamptz DEFAULT now() NULL,
 CONSTRAINT example_sink_t_pk PRIMARY KEY (id)
);
```

#### 查询语句

```SQL
SELECT * FROM example_data_t;
```

### influxdb

#### 创建数据

```SQL
CREATE DATABASE example;
```

#### 查询数据

```SQL
SELECT "value" FROM "example"."autogen"."example_data" WHERE time > :dashboardTime: AND time < :upperDashboardTime:
```

### 环境销毁

```shell
export HOST_IP=$(hostname -I | awk '{print $1}')
docker compose down
```

## 编译jar

```shell
mvn clean
mvn package
```

或

```shell
mvn clean package
```

## 参数说明

- --server.name 算子名称
- --execution.mode 执行模式，可选streaming、batch、automatic，大小写均可
- --influxdb.url influxdb数据库http地址，示例：<http://172.16.105.13:8888>
- --influxdb.database influxdb数据库名称，示例：HZW_TJZG_DL089_RING
- --influxdb.user influxdb数据库用户名，没有可不填
- --influxdb.password influxdb数据库密码，没有可不填
- --influxdb.batch.size 数据库批量写入大小，默认：100
- --influxdb.max.pool.size influxdb数据库连接池最大连接数，默认：10
- --influxdb.connection.timeout influxdb连接超时时间，默认：30000

### 服务器的flink jar包上传地址

```text
http://172.16.105.238:8081/#/overview
```

### 参数示例

```text
--server.name ringclean_htcjsd
--execution.mode streaming

--influxdb.url http://192.168.211.107:38086

--influxdb.user admin
--influxdb.password FZaStb0cXFuFbehPBM6YHCiuAAX6QIXr

--influxdb.database htcjsd_dz1368_ring
--influxdb.batch.size 100
--influxdb.max.pool.size 10
--influxdb.connection.timeout 300000
```

### run flink的顺序
1. 在服务器路径`/root/home/flink_htcjsd`上启动服务 `docker compose up -d` 或者重启服务 `docker compose restart`
2. 在flink_htcjsd项目中`mvn clean package` 打jar包
3. 上传flink_htcjsd的jar包到`http://172.16.105.238:8081/#/overview`
4. 获取flink_htcjsd的jar包id`curl http://localhost:8081/jars`
5. 在spring boot定时任务的`application.yml`中修改要上传jar包id，然后生成jar包
6. 上传jar包到服务器，然后启动命令运行 
   1. ``` text  
      docker run -d \
      --name algorithm-timer-app \
      --network="host" \
      -v /home/target/algorithmTimer-0.0.1-SNAPSHOT.jar:/app/application.jar \
      -w /app \
      openjdk:17 \
      java -jar application.jar [你的应用参数，如果有的话]

### jar包ID的获取
```text
curl http://localhost:8081/jars
```

### curl 运行jar包
```text
curl -X POST -H "Content-Type: application/json" -d \
'{
  "entryClass": "com.example.Example",
  "parallelism": 1,
  "programArgs": "--server.name example --execution.mode streaming --mysql.source.url jdbc:mysql://172.16.105.132:3306/ddg_tenant?characterEncoding=utf8&useSSL=false&serverTimezone=UTC&allowPublicKeyRetrieval=true --mysql.source.user ddg --mysql.source.password ddg.2021 --mysql.source.batch.size 100 --mysql.source.max.pool.size 10 --mysql.source.max.lifetime 1800000 --mysql.source.min.idle 5 --mysql.source.idle.timeout 30000 --mysql.source.connection.timeout 30000 --mysql.sink.url jdbc:mysql://172.16.105.105:3367/ddg?characterEncoding=utf8&useSSL=false&serverTimezone=UTC&allowPublicKeyRetrieval=true --mysql.sink.user root --mysql.sink.password jLw9cxdXJANxCQ68 --mysql.sink.batch.size 100 --mysql.sink.max.pool.size 10 --mysql.sink.max.lifetime 1800000 --mysql.sink.min.idle 5 --mysql.sink.idle.timeout 30000 --mysql.sink.connection.timeout 30000 --request.url http://172.16.105.238:8000/predict"
}' \
http://localhost:8081/jars/f552fe66-0f07-47fd-b49a-3a8da473056d_example-1.0.0-SNAPSHOT.jar/run
```

### 重启docker所有容器
重启位置
 cd /root/home/flink_htcjsd

```text
docker compose restart
```

启动所有服务
docker compose up -d


### 主驱动风险预警的job  
数据来源表： 
通苏嘉甬实时数据  

算法结果存储表：  
172.16.105.105  
ddg.risk_predict  

