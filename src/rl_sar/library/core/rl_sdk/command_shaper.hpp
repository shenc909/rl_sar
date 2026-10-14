/*
 * Copyright (c) 2024-2026 Shen Chen
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef COMMAND_SHAPER_HPP
#define COMMAND_SHAPER_HPP

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

// Preprocesses the operator velocity command [vx, vy, wz] before it becomes the policy's "commands"
// observation. Stages run in this order and each one is disabled when its config is empty:
//   1. deadband          |cmd| < deadband[i] -> 0 (stick drift around centre)
//   2. rate limit        per-axis slew limit, separate accel / decel rates (sharp jumps); stateful
//   3. tracking map      piecewise-linear desired -> commanded velocity (sim2real tracking gap)
//   4. range clamp       keep the command inside the policy's training range
//   5. lateral accel     |wz| <= max_lateral_accel / |(vx, vy)| (policies trained under this cap)
// Stages 1-2 work on the velocity the operator wants, so the rate limits are physical accelerations;
// stages 3-5 turn that into the command the policy is given.
class CommandShaper
{
public:
    using Curve = std::vector<std::pair<float, float>>;  // (desired, commanded), sorted by desired

    struct Config
    {
        std::vector<float> deadband;
        std::vector<float> max_accel;  // units/s; <= 0 leaves that axis unlimited
        std::vector<float> max_decel;  // used while |cmd| shrinks or the sign flips; empty = max_accel
        std::vector<Curve> tracking_map;  // per axis; an empty curve is identity
        std::vector<float> range_lower;
        std::vector<float> range_upper;
        float max_lateral_accel = 0.0f;
    };

    // Per-axis settings whose size is not num_axes are dropped (that stage is disabled), so Step can
    // index them freely; callers validate first to report the problem.
    void Configure(const Config& config, size_t num_axes)
    {
        this->config = config;
        this->num_axes = num_axes;
        Config& c = this->config;
        auto fits = [num_axes](const auto& v) { return v.size() == num_axes; };
        if (!fits(c.deadband)) c.deadband.clear();
        if (!fits(c.max_accel)) c.max_accel.clear();
        if (!fits(c.max_decel)) c.max_decel.clear();
        if (!fits(c.tracking_map)) c.tracking_map.clear();
        if (!fits(c.range_lower) || !fits(c.range_upper))
        {
            c.range_lower.clear();
            c.range_upper.clear();
        }
        this->Reset();
    }

    // Forget the rate limiter's state so the next Step starts from rest.
    void Reset()
    {
        this->limited.assign(this->num_axes, 0.0f);
        this->output.assign(this->num_axes, 0.0f);
    }

    bool Active() const
    {
        return !this->config.deadband.empty() || !this->config.max_accel.empty() ||
               !this->config.tracking_map.empty() || !this->config.range_lower.empty() ||
               this->config.max_lateral_accel > 0.0f;
    }

    // Advance one policy step of length dt and return the command for the policy.
    const std::vector<float>& Step(const std::vector<float>& raw, float dt)
    {
        if (raw.size() != this->num_axes)
        {
            this->Configure(this->config, raw.size());
        }
        const Config& c = this->config;
        const std::vector<float>& max_decel = c.max_decel.empty() ? c.max_accel : c.max_decel;

        for (size_t i = 0; i < this->num_axes; ++i)
        {
            float cmd = raw[i];
            if (!c.deadband.empty() && std::fabs(cmd) < c.deadband[i])
            {
                cmd = 0.0f;
            }
            if (!c.max_accel.empty())
            {
                cmd = RateLimit(this->limited[i], cmd, c.max_accel[i], max_decel[i], dt);
            }
            this->limited[i] = cmd;
            if (!c.tracking_map.empty())
            {
                cmd = Interpolate(c.tracking_map[i], cmd);
            }
            if (!c.range_lower.empty())
            {
                cmd = std::clamp(cmd, c.range_lower[i], c.range_upper[i]);
            }
            this->output[i] = cmd;
        }

        if (c.max_lateral_accel > 0.0f && this->num_axes >= 3)
        {
            // The yaw rate is clamped so the commanded speed is kept.
            const float speed = std::hypot(this->output[0], this->output[1]);
            if (speed > 0.0f)
            {
                const float max_yaw = c.max_lateral_accel / speed;
                this->output[2] = std::clamp(this->output[2], -max_yaw, max_yaw);
            }
        }
        return this->output;
    }

    // Last command handed to the policy (for display).
    const std::vector<float>& Output() const { return this->output; }

    // Build a tracking curve from (desired, commanded) points. (0, 0) is added unless a point sits at 0,
    // and a side with no points is mirrored from the other (odd symmetry), so a single [1.0, 1.15]
    // point is a plain 1.15 gain and forward-only calibration also covers walking backwards.
    // Returns an empty curve if two points share the same desired value.
    static Curve MakeCurve(const std::vector<std::vector<float>>& points)
    {
        Curve curve;
        bool has_neg = false, has_pos = false, has_zero = false;
        for (const auto& p : points)
        {
            if (p.size() != 2) return {};
            curve.emplace_back(p[0], p[1]);
            has_neg |= p[0] < 0.0f;
            has_pos |= p[0] > 0.0f;
            has_zero |= p[0] == 0.0f;
        }
        if (curve.empty()) return {};
        if (has_neg != has_pos)
        {
            const size_t n = curve.size();
            for (size_t i = 0; i < n; ++i)
            {
                if (curve[i].first != 0.0f) curve.emplace_back(-curve[i].first, -curve[i].second);
            }
        }
        if (!has_zero) curve.emplace_back(0.0f, 0.0f);
        std::sort(curve.begin(), curve.end());
        for (size_t i = 1; i < curve.size(); ++i)
        {
            if (curve[i].first == curve[i - 1].first) return {};
        }
        return curve;
    }

private:
    Config config;
    size_t num_axes = 0;
    std::vector<float> limited;  // rate-limited desired command, carried between steps
    std::vector<float> output;

    static float RateLimit(float prev, float target, float accel, float decel, float dt)
    {
        // Reversing direction brakes to zero first, at the decel rate.
        if (prev * target < 0.0f) target = 0.0f;
        const float rate = std::fabs(target) < std::fabs(prev) ? decel : accel;
        if (rate <= 0.0f) return target;
        const float max_step = rate * dt;
        return prev + std::clamp(target - prev, -max_step, max_step);
    }

    // Piecewise-linear lookup that extrapolates with the end segments' slopes.
    static float Interpolate(const Curve& curve, float x)
    {
        if (curve.size() < 2) return x;
        size_t i = 1;
        while (i < curve.size() - 1 && x > curve[i].first) ++i;
        const auto& [x0, y0] = curve[i - 1];
        const auto& [x1, y1] = curve[i];
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
    }
};

#endif // COMMAND_SHAPER_HPP
