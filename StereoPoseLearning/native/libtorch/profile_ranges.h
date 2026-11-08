#pragma once
// Disabled in the production build. Profiling uses build/profile separately.
#ifdef STEREO_PROFILE_RANGES
#include <ATen/record_function.h>
#include <memory>
#include <functional>
inline std::function<void(const char*)> stereo_stage_hook;
class ProfileStages {
    std::unique_ptr<at::RecordFunction> range_;
public:
    void next(const char* name) {
        range_.reset();
        if(stereo_stage_hook) stereo_stage_hook(name);
        range_=std::make_unique<at::RecordFunction>(at::RecordScope::USER_SCOPE);
        if(range_->isActive()) range_->before(name);
    }
    ~ProfileStages() { if(stereo_stage_hook) stereo_stage_hook("stage::__end__"); }
};
#else
struct ProfileStages { void next(const char*) {} };
#endif
