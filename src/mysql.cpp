#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>

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

void mysqlServer::insert_readings(std::vector<MeterInfo> &meters)
{
    // insert readings into Readings table
    for (auto &meter: meters)
    {
        if (meter.class_name == "N/A")
        {
            continue;
        }
        
        auto now = std::chrono::system_clock::now();
        uint64_t dis_millseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count()
            - std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count() * 1000;
        time_t tt = std::chrono::system_clock::to_time_t(now);
        auto time_tm = localtime(&tt);
        char cur_datetime[25] = { 0 };
        sprintf(cur_datetime, "%d%02d%02d-%02d%02d%02d.%03d", time_tm->tm_year + 1900,
            time_tm->tm_mon + 1, time_tm->tm_mday, time_tm->tm_hour,
            time_tm->tm_min, time_tm->tm_sec, (int)dis_millseconds);
        
        // DEBUG_PATH + current datetime
        std::string debug_image_path = DEBUG_PATH 
            + std::to_string(meter.camera_id) + "_" 
            + std::to_string(meter.instrument_id) + "_"
            + cur_datetime + "/";

        // create debug image path
        boost::filesystem::create_directories(debug_image_path);

        // save debug images
        if (meter.class_id == 0)
        {
            cv::imwrite(debug_image_path + "crop.jpg", meter.crop);
            cv::imwrite(debug_image_path + "mask_pointer.jpg", meter.mask_pointer);
            cv::imwrite(debug_image_path + "mask_scale.jpg", meter.mask_scale);
            cv::imwrite(debug_image_path + "circle.jpg", meter.circle);
            cv::imwrite(debug_image_path + "rect_pointer.jpg", meter.rect_pointer);
            cv::imwrite(debug_image_path + "rect_scale.jpg", meter.rect_scale);
        }
        else if (meter.class_id == 1)
        {
            cv::imwrite(debug_image_path + "crop.jpg", meter.crop);
        }

        // auto t1 = std::chrono::high_resolution_clock::now();
        std::string query = "INSERT INTO Readings (camera_id, instrument_id, value, datetime, rect_x, rect_y, rect_h, rect_w, debug_image_path) VALUES (" 
            + std::to_string(meter.camera_id) + ", " 
            + std::to_string(meter.instrument_id) + ", " 
            + std::to_string(meter.meter_reading_value) + ", NOW(3), " 
            + std::to_string(meter.rect.x) + ", " 
            + std::to_string(meter.rect.y) + ", " 
            + std::to_string(meter.rect.height) + ", " 
            + std::to_string(meter.rect.width) + ", '" 
            + DEBUG_PATH + "')";
        execute_query(this->stmt, query);
        // auto t2 = std::chrono::high_resolution_clock::now();
        // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

        // 

        LOG(INFO) << query;
        // LOG(WARNING) << "insert readings time: " << duration << " ms";
    }
}