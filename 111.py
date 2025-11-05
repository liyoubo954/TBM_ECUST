#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建四个风险数据表脚本
严格按照图片中的表结构创建四个风险表
"""

import pymysql
import sys
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MySQLTableCreator:
    def __init__(self, host, port, user, password, database):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.connection = None

    def connect(self):
        """连接MySQL数据库"""
        try:
            self.connection = pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )
            logger.info(f"成功连接到MySQL数据库: {self.host}:{self.port}/{self.database}")
            return True
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            return False

    def close(self):
        """关闭数据库连接"""
        if self.connection:
            self.connection.close()
            logger.info("数据库连接已关闭")

    def table_exists(self, table_name):
        """检查表是否存在"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(f"""
                    SELECT COUNT(*) as count 
                    FROM information_schema.tables 
                    WHERE table_schema = '{self.database}' 
                    AND table_name = '{table_name}'
                """)
                result = cursor.fetchone()
                return result['count'] > 0
        except Exception as e:
            logger.error(f"检查表是否存在失败: {e}")
            return False

    def drop_table_if_exists(self, table_name):
        """如果表存在则删除"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(f"DROP TABLE IF EXISTS `{table_name}`")
                self.connection.commit()
                logger.info(f"已删除存在的表: {table_name}")
                return True
        except Exception as e:
            logger.error(f"删除表失败: {e}")
            return False

    def create_table(self, table_name):
        """严格按照图片中的表结构创建数据表"""

        # 根据图片中的精确字段定义创建表
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS `{table_name}` (
            `id` INT NOT NULL AUTO_INCREMENT,
            `ring` INT,
            `project_name` VARCHAR(100),
            `risk_type` VARCHAR(20),
            `warning_time` VARCHAR(20),
            `warning_parameters` JSON,
            `safety_level` VARCHAR(20),
            `risk_level` VARCHAR(20),
            `risk_score` DECIMAL(5,2),
            `potential_risk` TEXT,
            `fault_reason` TEXT,
            `fault_measures` TEXT,
            `fault_cause` TEXT,
            `impact_parameters` TEXT,
            `created_time` DATETIME,

            PRIMARY KEY (`id`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(create_table_sql)
                self.connection.commit()
                logger.info(f"表 {table_name} 创建成功")
                return True
        except Exception as e:
            logger.error(f"创建表 {table_name} 失败: {e}")
            self.connection.rollback()
            return False

    def create_all_tables(self, drop_existing=False):
        """创建四个数据表"""
        tables = ['clog_risk', 'mud_cake_risk', 'tail_seal_risk', 'mdr_seal_risk']

        for table_name in tables:
            if drop_existing:
                self.drop_table_if_exists(table_name)
            elif self.table_exists(table_name):
                logger.info(f"表 {table_name} 已存在，跳过创建")
                continue

            if not self.create_table(table_name):
                logger.error(f"创建表 {table_name} 失败，终止执行")
                return False

        logger.info("所有表创建完成")
        return True

    def show_table_structure(self, table_name):
        """显示表结构"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(f"DESCRIBE {table_name}")
                result = cursor.fetchall()
                logger.info(f"表 {table_name} 结构:")
                print(f"\n{table_name} 表结构:")
                print("-" * 80)
                print(f"{'字段名':<20} {'类型':<15} {'是否为空':<10} {'键':<10} {'默认值':<15} {'额外信息':<10}")
                print("-" * 80)

                for column in result:
                    field = column['Field']
                    field_type = column['Type']
                    is_null = 'YES' if column['Null'] == 'YES' else 'NO'
                    key = column['Key'] if column['Key'] else ''
                    default = str(column['Default']) if column['Default'] is not None else 'NULL'
                    extra = column['Extra'] if column['Extra'] else ''

                    print(f"{field:<20} {field_type:<15} {is_null:<10} {key:<10} {default:<15} {extra:<10}")

                return result
        except Exception as e:
            logger.error(f"获取表结构失败: {e}")
            return None

    def validate_table_structure(self, table_name):
        """验证表结构是否符合图片标准"""
        expected_columns = {
            'id': {'type': 'int', 'null': 'NO', 'key': 'PRI'},
            'ring': {'type': 'int', 'null': 'YES', 'key': ''},
            'project_name': {'type': 'varchar(100)', 'null': 'YES', 'key': ''},
            'risk_type': {'type': 'varchar(20)', 'null': 'YES', 'key': ''},
            'warning_time': {'type': 'varchar(20)', 'null': 'YES', 'key': ''},
            'warning_parameters': {'type': 'json', 'null': 'YES', 'key': ''},
            'safety_level': {'type': 'varchar(20)', 'null': 'YES', 'key': ''},
            'risk_level': {'type': 'varchar(20)', 'null': 'YES', 'key': ''},
            'risk_score': {'type': 'decimal(5,2)', 'null': 'YES', 'key': ''},
            'potential_risk': {'type': 'text', 'null': 'YES', 'key': ''},
            'fault_reason': {'type': 'text', 'null': 'YES', 'key': ''},
            'fault_measures': {'type': 'text', 'null': 'YES', 'key': ''},
            'fault_cause': {'type': 'text', 'null': 'YES', 'key': ''},
            'impact_parameters': {'type': 'text', 'null': 'YES', 'key': ''},
            'created_time': {'type': 'datetime', 'null': 'YES', 'key': ''}
        }

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(f"DESCRIBE {table_name}")
                actual_columns = cursor.fetchall()

                logger.info(f"验证表 {table_name} 结构...")
                validation_passed = True

                for expected_col, expected_props in expected_columns.items():
                    found = False
                    for actual_col in actual_columns:
                        if actual_col['Field'] == expected_col:
                            found = True
                            actual_type = actual_col['Type'].lower()
                            expected_type = expected_props['type'].lower()

                            # 验证类型
                            if expected_type not in actual_type:
                                logger.warning(
                                    f"字段 {expected_col} 类型不匹配: 期望 {expected_type}, 实际 {actual_type}")
                                validation_passed = False

                            # 验证是否允许空值
                            expected_null = expected_props['null']
                            actual_null = actual_col['Null']
                            if expected_null != actual_null:
                                logger.warning(
                                    f"字段 {expected_col} 空值设置不匹配: 期望 {expected_null}, 实际 {actual_null}")
                                validation_passed = False

                            break

                    if not found:
                        logger.error(f"缺少字段: {expected_col}")
                        validation_passed = False

                if validation_passed:
                    logger.info(f"✅ 表 {table_name} 结构验证通过")
                else:
                    logger.error(f"❌ 表 {table_name} 结构验证失败")

                return validation_passed

        except Exception as e:
            logger.error(f"验证表结构失败: {e}")
            return False


def main():
    """主函数"""

    # 数据库连接配置
    db_config = {
        'host': '192.168.211.104',
        'port': 6446,
        'user': 'root',
        'password': '7m@9X!zP2qA5LbNcRfTgYhJkM3nD4v6B',
        'database': 'algorithm'
    }

    # 创建表创建器实例
    creator = MySQLTableCreator(**db_config)

    try:
        # 连接数据库
        if not creator.connect():
            sys.exit(1)

        # 创建四个表（如果已存在则删除重建）
        if creator.create_all_tables(drop_existing=True):
            logger.info("✅ 所有数据表创建任务完成")

            # 显示表结构
            for table_name in ['clog_risk', 'mud_cake_risk', 'tail_seal_risk', 'mdr_seal_risk']:
                creator.show_table_structure(table_name)
                creator.validate_table_structure(table_name)
        else:
            logger.error("❌ 数据表创建任务失败")
            sys.exit(1)

    except Exception as e:
        logger.error(f"执行过程中发生错误: {e}")
        sys.exit(1)
    finally:
        creator.close()


if __name__ == "__main__":
    main()