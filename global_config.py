import time

# 日志文件
LOG_FILENAME = './logs/log_%s.txt' % time.strftime('%Y%m%d')

# 日志格式
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# 日志日期格式
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
