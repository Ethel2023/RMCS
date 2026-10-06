#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numbers>
#include <string>

#include <librmcs/board/c_board.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/node_options.hpp>
#include <rmcs_executor/component.hpp>
#include <rmcs_msgs/switch.hpp>

#include "controller/pid/pid_calculator.hpp"
#include "filter/low_pass_filter.hpp"
#include "hardware/device/can_packet.hpp"
#include "hardware/device/dji_motor.hpp"
#include "hardware/device/dr16.hpp"

namespace rmcs_core::hardware {

class GantryLauncherTest
    : public rmcs_executor::Component
    , public rclcpp::Node
    , public librmcs::board::CBoard::Callback {
public:
    GantryLauncherTest()
        : Node(
              get_component_name(),
              rclcpp::NodeOptions{}.automatically_declare_parameters_from_overrides(true))
        , command_(
              create_partner_component<GantryCommand>(get_component_name() + "_command", *this))
        , pitch_left_motor_(*this, *this, "/gantry/pitch_left")
        , pitch_right_motor_(*this, *this, "/gantry/pitch_right")
        , yaw_motor_(*this, *this, "/gantry/yaw")
        , pitch_velocity_filter_left_(
              get_parameter("filter_cutoff_hz").as_double(),
              get_parameter("filter_sampling_hz").as_double())
        , pitch_velocity_filter_right_(
              get_parameter("filter_cutoff_hz").as_double(),
              get_parameter("filter_sampling_hz").as_double())
        , yaw_velocity_filter_(
              get_parameter("filter_cutoff_hz").as_double(),
              get_parameter("filter_sampling_hz").as_double())
        , pitch_rate_max_(get_parameter("pitch_rate_max").as_double())
        , pitch_velocity_feedforward_(get_parameter("pitch_velocity_feedforward").as_double())
        , yaw_rate_max_(get_parameter("yaw_rate_max").as_double())
        , yaw_velocity_feedforward_(get_parameter("yaw_velocity_feedforward").as_double())
        , pitch_speed_limit_(get_parameter("pitch_speed_limit").as_double())
        , yaw_speed_limit_(get_parameter("yaw_speed_limit").as_double())
        , pitch_upper_limit_(get_parameter("pitch_upper_limit").as_double())
        , pitch_lower_limit_(get_parameter("pitch_lower_limit").as_double())
        , pitch_limit_enabled_(get_parameter("pitch_limit_enabled").as_bool())
        , skew_kp_(get_parameter("skew_kp").as_double())
        , deadzone_(get_parameter("deadzone").as_double())
        , dt_(1.0 / get_parameter("filter_sampling_hz").as_double()) {

        auto configure_motor =
            [](device::DjiMotor& motor, std::uint8_t id, bool reversed) {
                device::DjiMotor::Config config{device::DjiMotor::Type::kM2006, id};
                config.enable_multi_turn_angle();
                if (reversed)
                    config.set_reversed();
                motor.configure(config);
            };

        // ID 实际分配：yaw=1，pitch_right=2，pitch_left=3。
        configure_motor(
            pitch_left_motor_, 3, get_parameter("pitch_left_reversed").as_bool());
        configure_motor(
            pitch_right_motor_, 2, get_parameter("pitch_right_reversed").as_bool());
        configure_motor(yaw_motor_, 1, get_parameter("yaw_reversed").as_bool());

        auto setup_pid = [](
                             controller::pid::PidCalculator& pid, double kp, double ki, double kd,
                             double output_limit, double integral_limit) {
            pid.kp = kp;
            pid.ki = ki;
            pid.kd = kd;
            pid.output_max = output_limit;
            pid.output_min = -output_limit;
            pid.integral_max = integral_limit;
            pid.integral_min = -integral_limit;
            pid.reset();
        };

        const double pitch_position_kp = get_parameter("pitch_position_kp").as_double();
        const double pitch_position_ki = get_parameter("pitch_position_ki").as_double();
        const double pitch_position_kd = get_parameter("pitch_position_kd").as_double();
        const double pitch_velocity_kp = get_parameter("pitch_velocity_kp").as_double();
        const double pitch_velocity_ki = get_parameter("pitch_velocity_ki").as_double();
        const double pitch_velocity_kd = get_parameter("pitch_velocity_kd").as_double();

        const double yaw_position_kp = get_parameter("yaw_position_kp").as_double();
        const double yaw_position_ki = get_parameter("yaw_position_ki").as_double();
        const double yaw_position_kd = get_parameter("yaw_position_kd").as_double();
        const double yaw_velocity_kp = get_parameter("yaw_velocity_kp").as_double();
        const double yaw_velocity_ki = get_parameter("yaw_velocity_ki").as_double();
        const double yaw_velocity_kd = get_parameter("yaw_velocity_kd").as_double();

        const double position_integral_limit = get_parameter("position_integral_limit").as_double();
        const double velocity_integral_limit = get_parameter("velocity_integral_limit").as_double();

        setup_pid(
            pitch_position_pid_left_, pitch_position_kp, pitch_position_ki, pitch_position_kd,
            pitch_speed_limit_, position_integral_limit);
        setup_pid(
            pitch_position_pid_right_, pitch_position_kp, pitch_position_ki, pitch_position_kd,
            pitch_speed_limit_, position_integral_limit);
        setup_pid(
            pitch_velocity_pid_left_, pitch_velocity_kp, pitch_velocity_ki, pitch_velocity_kd,
            pitch_left_motor_.max_torque(), velocity_integral_limit);
        setup_pid(
            pitch_velocity_pid_right_, pitch_velocity_kp, pitch_velocity_ki, pitch_velocity_kd,
            pitch_right_motor_.max_torque(), velocity_integral_limit);

        setup_pid(
            yaw_position_pid_, yaw_position_kp, yaw_position_ki, yaw_position_kd,
            yaw_speed_limit_, position_integral_limit);
        setup_pid(
            yaw_velocity_pid_, yaw_velocity_kp, yaw_velocity_ki, yaw_velocity_kd,
            yaw_motor_.max_torque(), velocity_integral_limit);

        board_ = std::make_unique<librmcs::board::CBoard>(
            *this, get_parameter("board_serial").as_string());
    }

    void update() override {
        dr16_.update_status();
        pitch_left_motor_.update_status();
        pitch_right_motor_.update_status();
        yaw_motor_.update_status();

        //遥控器掉线检测
        if (!dr16_.valid() || emergency_stop_requested()) {
            reset_controllers();
            pitch_left_torque_.store(0.0);
            pitch_right_torque_.store(0.0);
            yaw_torque_.store(0.0);
            return;
        }

        //初始化检测放飞车
        if (!targets_initialized_) {
            pitch_target_ = 0.5 * (pitch_left_motor_.angle() + pitch_right_motor_.angle());
            yaw_target_ = yaw_motor_.angle();
            targets_initialized_ = true;
        }

        // 左杆上下：pitch；左杆左右：yaw。
        const double pitch_cmd = apply_deadzone(-dr16_.joystick_left().y(), deadzone_);
        const double yaw_cmd = apply_deadzone(dr16_.joystick_left().x(), deadzone_);
        // 计算目标角度，不用纯摇杆映射
        pitch_target_ += pitch_cmd * pitch_rate_max_ * dt_;
        if (pitch_limit_enabled_)
            pitch_target_ = std::clamp(pitch_target_, pitch_lower_limit_, pitch_upper_limit_);
        yaw_target_ += yaw_cmd * yaw_rate_max_ * dt_;

        // 读取左右 pitch 电机角度
        const double left_angle = pitch_left_motor_.angle();
        const double right_angle = pitch_right_motor_.angle();

        // 速度滤波
        const double left_velocity =
            pitch_velocity_filter_left_.update(pitch_left_motor_.velocity());
        const double right_velocity =
            pitch_velocity_filter_right_.update(pitch_right_motor_.velocity());
        // 速度前馈
        const double pitch_velocity_feedforward =
            pitch_velocity_feedforward_ * pitch_cmd * pitch_rate_max_;

        // 计算输出扭矩
        double left_torque = pitch_velocity_pid_left_.update(
            pitch_position_pid_left_.update(pitch_target_ - left_angle)
            + pitch_velocity_feedforward - left_velocity);
        double right_torque = pitch_velocity_pid_right_.update(
            pitch_position_pid_right_.update(pitch_target_ - right_angle)
            + pitch_velocity_feedforward - right_velocity);

        // 左右 pitch 同步修正，防止升降歪斜。
        const double pitch_skew = left_angle - right_angle;
        left_torque -= skew_kp_ * pitch_skew;
        right_torque += skew_kp_ * pitch_skew;

        // yaw 独立不参与 pitch 计算。
        const double yaw_error = wrap_angle_error(yaw_target_ - yaw_motor_.angle());
        const double yaw_velocity = yaw_velocity_filter_.update(yaw_motor_.velocity());
        const double yaw_velocity_reference =
            yaw_position_pid_.update(yaw_error) + yaw_velocity_feedforward_ * yaw_cmd * yaw_rate_max_;
        const double yaw_torque = yaw_velocity_pid_.update(yaw_velocity_reference - yaw_velocity);

        pitch_left_torque_.store(left_torque);
        pitch_right_torque_.store(right_torque);
        yaw_torque_.store(yaw_torque);
    }

    void command_update() {
        auto builder = board_->start_transmit();

        device::CanPacket8 packet{
            device::CanPacket8::PaddingQuarter{},
            device::CanPacket8::PaddingQuarter{},
            device::CanPacket8::PaddingQuarter{},
            device::CanPacket8::PaddingQuarter{},
        };

        // M2006 控制帧 data 下标 = ID - 1。
        packet.data[0] = yaw_motor_.generate_command(yaw_torque_.load()).data;
        packet.data[1] = pitch_right_motor_.generate_command(pitch_right_torque_.load()).data;
        packet.data[2] = pitch_left_motor_.generate_command(pitch_left_torque_.load()).data;

        // yaw 电机 ID = 1，pitch_right 电机 ID = 2，pitch_left 电机 ID = 3。
        builder.can_transmit(
            Spec::kCans.kCan1,
            {.can_id = 0x200, .can_data = packet.as_bytes()});
    }

private:
    // 摇杆死区检测
    static double apply_deadzone(double value, double deadzone) {
        if (std::abs(value) < deadzone)
            return 0.0;
        if (value > 0.0)
            return (value - deadzone) / (1.0 - deadzone);
        return (value + deadzone) / (1.0 - deadzone);
    }

    // 角度误差化归
    static double wrap_angle_error(double error) {
        constexpr double two_pi = 2.0 * std::numbers::pi;
        error = std::fmod(error + std::numbers::pi, two_pi);
        if (error < 0.0)
            error += two_pi;
        return error - std::numbers::pi;
    }

    void reset_controllers() {
        pitch_position_pid_left_.reset();
        pitch_position_pid_right_.reset();
        pitch_velocity_pid_left_.reset();
        pitch_velocity_pid_right_.reset();
        yaw_position_pid_.reset();
        yaw_velocity_pid_.reset();

        pitch_velocity_filter_left_.reset();
        pitch_velocity_filter_right_.reset();
        yaw_velocity_filter_.reset();

        targets_initialized_ = false;
    }

    // DR16 两个拨杆同时处于 DOWN 时所有电机电流清零。
    bool emergency_stop_requested() const {
        return dr16_.switch_left() == rmcs_msgs::Switch::DOWN
            && dr16_.switch_right() == rmcs_msgs::Switch::DOWN;
    }

    // CAN 接收回调
    void can_receive_callback(const Spec::Can& can, const View::Can& data) override {
        if (can == Spec::kCans.kCan1) {
            pitch_left_motor_.match_then_store_status(data.can_id, data.can_data);
            pitch_right_motor_.match_then_store_status(data.can_id, data.can_data);
            yaw_motor_.match_then_store_status(data.can_id, data.can_data);
        }
    }

    // UART 接收回调
    void uart_receive_callback(const Spec::Uart& uart, const View::Uart& data) override {
        if (uart == Spec::kUarts.kDbus)
            dr16_.store_status(data.uart_data.data(), data.uart_data.size());
    }
    // 空回调
    void gpio_digital_read_result_callback(const Spec::Gpio&, const View::GpioDigital&) override {}
    void accelerometer_receive_callback(const View::ImuAccelerometer&) override {}
    void gyroscope_receive_callback(const View::ImuGyroscope&) override {}

    class GantryCommand : public rmcs_executor::Component {
    public:
        explicit GantryCommand(GantryLauncherTest& gantry)
            : gantry_(gantry) {}

        void update() override { gantry_.command_update(); }

    private:
        GantryLauncherTest& gantry_;
    };

    std::shared_ptr<GantryCommand> command_;

    device::DjiMotor pitch_left_motor_;
    device::DjiMotor pitch_right_motor_;
    device::DjiMotor yaw_motor_;
    device::Dr16 dr16_;

    controller::pid::PidCalculator pitch_position_pid_left_;
    controller::pid::PidCalculator pitch_position_pid_right_;
    controller::pid::PidCalculator pitch_velocity_pid_left_;
    controller::pid::PidCalculator pitch_velocity_pid_right_;
    controller::pid::PidCalculator yaw_position_pid_;
    controller::pid::PidCalculator yaw_velocity_pid_;

    filter::LowPassFilter<1> pitch_velocity_filter_left_;
    filter::LowPassFilter<1> pitch_velocity_filter_right_;
    filter::LowPassFilter<1> yaw_velocity_filter_;

    std::atomic<double> pitch_left_torque_{0.0};
    std::atomic<double> pitch_right_torque_{0.0};
    std::atomic<double> yaw_torque_{0.0};

    double pitch_target_ = 0.0;
    double yaw_target_ = 0.0;
    bool targets_initialized_ = false;

    const double pitch_rate_max_;
    const double pitch_velocity_feedforward_;
    const double yaw_rate_max_;
    const double yaw_velocity_feedforward_;
    const double pitch_speed_limit_;
    const double yaw_speed_limit_;
    const double pitch_upper_limit_;
    const double pitch_lower_limit_;
    const bool pitch_limit_enabled_;
    const double skew_kp_;
    const double deadzone_;
    const double dt_;

    std::unique_ptr<librmcs::board::CBoard> board_;
};

} // namespace rmcs_core::hardware

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(rmcs_core::hardware::GantryLauncherTest, rmcs_executor::Component)
