import time
import subprocess
import logging
import re
from prometheus_client import start_http_server, Gauge

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("system_exporter.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("system_exporter")

# Defining Prometheus metrics for IO statistics
io_read_rate = Gauge('io_read_rate', 'IO read rate', ['device'])
io_write_rate = Gauge('io_write_rate', 'IO write rate', ['device'])
io_tps = Gauge('io_tps', 'IO transactions per second', ['device'])
io_read_bytes = Gauge('io_read_bytes', 'IO read bytes', ['device'])
io_write_bytes = Gauge('io_write_bytes', 'IO write bytes', ['device'])

# Defining Prometheus metrics for CPU statistics
cpu_avg_percent = Gauge('cpu_avg_percent', 'CPU average percentage', ['mode'])

# Defining Prometheus metrics for memory statistics
mem_info = Gauge('mem_info', 'Memory information in kB', ['type'])


def collect_iostat_metrics():
    """
    Collects IO and CPU statistics using iostat command and
    updates Prometheus metric objects.
    """
    logger.info("Collecting IO and CPU metrics")
    
    try:
        # Runnig iostat command and capture output
        process = subprocess.Popen(['iostat'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        if stderr:
            logger.error(f"Error running iostat: {stderr.decode('utf-8')}")
            return
            
        output = stdout.decode('utf-8')
        logger.debug(f"iostat output: {output}")
        
        # Parse CPU statistics
        cpu_pattern = r'avg-cpu:\s+%user\s+%nice\s+%system\s+%iowait\s+%steal\s+%idle\s*\n\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)'
        cpu_match = re.search(cpu_pattern, output)
        
        if cpu_match:
            user, nice, system, iowait, steal, idle = map(float, cpu_match.groups())
            cpu_avg_percent.labels(mode='user').set(user)
            cpu_avg_percent.labels(mode='nice').set(nice)
            cpu_avg_percent.labels(mode='system').set(system)
            cpu_avg_percent.labels(mode='iowait').set(iowait)
            cpu_avg_percent.labels(mode='steal').set(steal)
            cpu_avg_percent.labels(mode='idle').set(idle)
            logger.info("Updated CPU metrics")
        else:
            logger.warning("Could not parse CPU statistics from iostat output")
        
        # Parse device IO statistics
        lines = output.strip().split('\n')
        device_section_start = False
        
        for line in lines:
            if 'Device' in line and 'tps' in line:
                device_section_start = True
                continue
                
            if device_section_start and line.strip():
                parts = line.split()
                if len(parts) >= 6:
                    device = parts[0]
                    tps_val = float(parts[1])
                    read_kb = float(parts[2])
                    write_kb = float(parts[3])
                    
                    # Update Prometheus metrics
                    io_tps.labels(device=device).set(tps_val)
                    io_read_rate.labels(device=device).set(read_kb)
                    io_write_rate.labels(device=device).set(write_kb)
                    io_read_bytes.labels(device=device).set(read_kb * 1024)  # Convert to bytes
                    io_write_bytes.labels(device=device).set(write_kb * 1024)  # Convert to bytes
                    
                    logger.info(f"Updated IO metrics for device {device}")
        
    except Exception as e:
        logger.error(f"Error collecting iostat metrics: {str(e)}")


def collect_memory_metrics():
    """
    Collects memory information from /proc/meminfo and
    updates Prometheus metric objects.
    """
    logger.info("Collecting memory metrics")
    
    try:
        # Read memory info from /proc/meminfo
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
        
        logger.debug(f"meminfo output: {meminfo}")
        
        # Parse memory statistics
        for line in meminfo.strip().split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value_parts = value.strip().split()
                
                if len(value_parts) >= 1:
                    try:
                        value_kb = float(value_parts[0])
                        mem_info.labels(type=key).set(value_kb)
                        logger.debug(f"Set memory metric {key} to {value_kb}")
                    except ValueError:
                        logger.warning(f"Could not parse value for {key}: {value_parts[0]}")
        
        logger.info("Updated memory metrics")
            
    except Exception as e:
        logger.error(f"Error collecting memory metrics: {str(e)}")


def main():
    """
    Main function to start HTTP server and collect metrics periodically.
    """
    # Start HTTP server for Prometheus scraping
    start_http_server(18000)
    logger.info("Started HTTP server on port 18000")
    
    # Collect metrics at regular intervals
    while True:
        collect_iostat_metrics()
        collect_memory_metrics()
        time.sleep(1)  # Collect metrics every second


if __name__ == "__main__":
    logger.info("Starting system metrics exporter")
    main()