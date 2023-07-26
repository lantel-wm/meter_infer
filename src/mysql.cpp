#include <opencv2/opencv.hpp>


#include "mysql.hpp"

void execute_query(std::auto_ptr<sql::Statement> &stmt, std::string query)
{
    try
    {
        stmt->execute(query);
    }
    catch (sql::SQLException &e)
    {
        LOG(ERROR) << "# ERR: SQLException in " << __FILE__;
        LOG(ERROR) << "(" << __FUNCTION__ << ") on line " << __LINE__;
        LOG(ERROR) << "# ERR: " << e.what();
        LOG(ERROR) << " (MySQL error code: " << e.getErrorCode();
        LOG(ERROR) << ", SQLState: " << e.getSQLState() << ")";
        LOG(ERROR) << "query: " << query;

        exit(EXIT_FAILURE);
    }
}

mysqlServer::mysqlServer(std::string sql_url, std::string user, std::string passwd, std::string database)
    : url(sql_url), user(user), passwd(passwd), database(database)
{
    try
    {
        this->driver = get_driver_instance();
        this->con = std::auto_ptr<sql::Connection>(driver->connect(url, user, passwd));
        this->con->setSchema(database);
        this->stmt = std::auto_ptr<sql::Statement>(con->createStatement());
    }
    catch (sql::SQLException &e)
    {
        LOG(ERROR) << "# ERR: SQLException in " << __FILE__;
        LOG(ERROR) << "(" << __FUNCTION__ << ") on line " << __LINE__;
        LOG(ERROR) << "# ERR: " << e.what();
        LOG(ERROR) << " (MySQL error code: " << e.getErrorCode();
        LOG(ERROR) << ", SQLState: " << e.getSQLState() << ")";

        exit(EXIT_FAILURE);
    }
}

mysqlServer::~mysqlServer()
{
    this->con->close();
}

void mysqlServer::init_camera_instruments(std::vector<FrameInfo> &frame_batch, std::vector<std::string> &urls)
{
    // delete all rows in Cameras table and Instruments table
    LOG(INFO) << "DELETE FROM Cameras";
    LOG(INFO) << "DELETE FROM Instruments";
    execute_query(this->stmt, "SET foreign_key_checks = 0");
    execute_query(this->stmt, "DELETE FROM Cameras");
    execute_query(this->stmt, "DELETE FROM Instruments");
    execute_query(this->stmt, "SET foreign_key_checks = 1");

    for (int camera_id = 0; camera_id < frame_batch.size(); camera_id++)
    {
        // insert cameras into Cameras table
        std::string query = "INSERT INTO Cameras (camera_id, camera_url) VALUES (" 
            + std::to_string(camera_id) + ", '" + urls[camera_id] + "')";
        LOG(INFO) << query;
        execute_query(this->stmt, query);
        

        // insert instruments into Instruments table
        for (auto &obj: frame_batch[camera_id].det_objs)
        {
            int class_id = obj.class_id;
            std::string unit = METER_UNITS[class_id];
            std::string query = "INSERT INTO Instruments (instrument_id, instrument_type, instrument_unit, camera_id) VALUES (" 
                + std::to_string(obj.instrument_id) 
                + ", '" + obj.class_name + "', '" + unit + "', " 
                + std::to_string(camera_id) + ")";
            LOG(INFO) << query;
            execute_query(this->stmt, query);
            
        }
    }
}

void mysqlServer::insert_readings(std::vector<MeterInfo> &meters, std::vector<std::chrono::steady_clock::time_point> &last_save_times)
{
    // insert readings into Readings table
    for (auto &meter: meters)
    {
        std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(now - last_save_times[meter.instrument_id]);
        
        if (time_span.count() < 1.0)
        {
            continue;
        }

        last_save_times[meter.instrument_id] = now;

        std::string query = "INSERT INTO Readings (camera_id, instrument_id, value, datetime, rect_x, rect_y, rect_h, rect_w, debug_image_path) VALUES (" 
            + std::to_string(meter.camera_id) + ", " 
            + std::to_string(meter.instrument_id) + ", " 
            + std::to_string(meter.meter_reading_value) + ", NOW(3), " 
            + std::to_string(meter.rect.x) + ", " 
            + std::to_string(meter.rect.y) + ", " 
            + std::to_string(meter.rect.height) + ", " 
            + std::to_string(meter.rect.width) + ", '" 
            + "no debug img" + "')";
        execute_query(this->stmt, query);
        LOG(INFO) << query;
    }
}