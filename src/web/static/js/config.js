/**
 * Configuration page JavaScript
 */

let currentConfig = {};

// Initialize configuration page
async function initConfig() {
    setupNavigation();
    setupRangeSliders();
    await loadConfiguration();
}

// Setup section navigation
function setupNavigation() {
    const navItems = document.querySelectorAll('#configNav .list-group-item');

    navItems.forEach(item => {
        item.addEventListener('click', function(e) {
            e.preventDefault();

            // Update active state
            navItems.forEach(nav => nav.classList.remove('active'));
            this.classList.add('active');

            // Show selected section
            const section = this.dataset.section;
            showSection(section);
        });
    });
}

// Show specific configuration section
function showSection(sectionName) {
    const sections = document.querySelectorAll('.config-section');

    sections.forEach(section => {
        if (section.id === `section-${sectionName}`) {
            section.classList.remove('d-none');
        } else {
            section.classList.add('d-none');
        }
    });
}

// Setup range slider value displays
function setupRangeSliders() {
    const sliders = document.querySelectorAll('input[type="range"]');

    sliders.forEach(slider => {
        const valueDisplay = document.getElementById(slider.id + '_value');

        slider.addEventListener('input', function() {
            if (valueDisplay) {
                valueDisplay.textContent = parseFloat(this.value).toFixed(2);
            }
        });
    });
}

// Load configuration from API
async function loadConfiguration() {
    try {
        const response = await apiCall('/api/config/');
        currentConfig = response.config;

        populateForm(currentConfig);
        showToast('Configuration loaded', 'success');

    } catch (error) {
        console.error('Failed to load configuration:', error);
        showToast('Failed to load configuration', 'error');
    }
}

// Populate form with configuration values
function populateForm(config) {
    // System settings
    setFormValue('system_name', config.system?.name);
    setFormValue('system_log_level', config.system?.log_level);

    // Tracking settings
    setFormValue('tracking_max_age', config.tracking?.max_age);
    setFormValue('tracking_min_hits', config.tracking?.min_hits);
    setFormValue('tracking_iou_threshold', config.tracking?.iou_threshold);

    // Pose model settings
    setFormValue('pose_model_model_path', config.pose_model?.model_path);
    setFormValue('pose_model_device', config.pose_model?.device);
    setFormValue('pose_model_conf_threshold', config.pose_model?.conf_threshold);

    // Fall detection settings
    setFormValue('fall_detection_enabled', config.fall_detection?.enabled, 'checkbox');
    setFormValue('fall_detection_confidence_threshold', config.fall_detection?.confidence_threshold);
    setFormValue('fall_detection_alert_cooldown_seconds', config.fall_detection?.alert_cooldown_seconds);
    setFormValue('fall_detection_vertical_threshold', config.fall_detection?.vertical_threshold);
    setFormValue('fall_detection_angle_threshold', config.fall_detection?.angle_threshold);

    // Bullying detection settings
    setFormValue('bullying_detection_enabled', config.bullying_detection?.enabled, 'checkbox');
    setFormValue('bullying_detection_confidence_threshold', config.bullying_detection?.confidence_threshold);
    setFormValue('bullying_detection_group_distance_threshold', config.bullying_detection?.group_distance_threshold);
    setFormValue('bullying_detection_group_min_size', config.bullying_detection?.group_min_size);
    setFormValue('bullying_detection_rapid_movement_threshold', config.bullying_detection?.rapid_movement_threshold);

    // POSH detection settings
    setFormValue('posh_detection_enabled', config.posh_detection?.enabled, 'checkbox');
    setFormValue('posh_detection_confidence_threshold', config.posh_detection?.confidence_threshold);
    setFormValue('posh_detection_isolation_distance_threshold', config.posh_detection?.isolation_distance_threshold);
    setFormValue('posh_detection_proximity_threshold', config.posh_detection?.proximity_threshold);
    setFormValue('posh_detection_prolonged_interaction_frames', config.posh_detection?.prolonged_interaction_frames);

    // Alert settings
    setFormValue('alerts_save_to_database', config.alerts?.save_to_database, 'checkbox');
    setFormValue('alerts_save_video_clips', config.alerts?.save_video_clips, 'checkbox');
    setFormValue('alerts_clip_duration_seconds', config.alerts?.clip_duration_seconds);
    setFormValue('alerts_max_alerts_per_minute', config.alerts?.max_alerts_per_minute);

    // Privacy settings
    setFormValue('privacy_blur_faces', config.privacy?.blur_faces, 'checkbox');
    setFormValue('privacy_save_only_pose_skeleton', config.privacy?.save_only_pose_skeleton, 'checkbox');
    setFormValue('privacy_retention_days', config.privacy?.retention_days);

    // Trigger range slider updates
    document.querySelectorAll('input[type="range"]').forEach(slider => {
        slider.dispatchEvent(new Event('input'));
    });
}

// Set form field value
function setFormValue(id, value, type = 'text') {
    const element = document.getElementById(id);

    if (!element || value === undefined) return;

    if (type === 'checkbox') {
        element.checked = value;
    } else {
        element.value = value;
    }
}

// Get form field value
function getFormValue(id, type = 'text') {
    const element = document.getElementById(id);

    if (!element) return undefined;

    if (type === 'checkbox') {
        return element.checked;
    } else if (type === 'number') {
        return parseFloat(element.value);
    } else {
        return element.value;
    }
}

// Save configuration
async function saveConfiguration() {
    try {
        // Build updates object
        const updates = {
            // System settings
            'system.name': getFormValue('system_name'),
            'system.log_level': getFormValue('system_log_level'),

            // Tracking settings
            'tracking.max_age': getFormValue('tracking_max_age', 'number'),
            'tracking.min_hits': getFormValue('tracking_min_hits', 'number'),
            'tracking.iou_threshold': getFormValue('tracking_iou_threshold', 'number'),

            // Pose model settings
            'pose_model.model_path': getFormValue('pose_model_model_path'),
            'pose_model.device': getFormValue('pose_model_device'),
            'pose_model.conf_threshold': getFormValue('pose_model_conf_threshold', 'number'),

            // Fall detection settings
            'fall_detection.enabled': getFormValue('fall_detection_enabled', 'checkbox'),
            'fall_detection.confidence_threshold': getFormValue('fall_detection_confidence_threshold', 'number'),
            'fall_detection.alert_cooldown_seconds': getFormValue('fall_detection_alert_cooldown_seconds', 'number'),
            'fall_detection.vertical_threshold': getFormValue('fall_detection_vertical_threshold', 'number'),
            'fall_detection.angle_threshold': getFormValue('fall_detection_angle_threshold', 'number'),

            // Bullying detection settings
            'bullying_detection.enabled': getFormValue('bullying_detection_enabled', 'checkbox'),
            'bullying_detection.confidence_threshold': getFormValue('bullying_detection_confidence_threshold', 'number'),
            'bullying_detection.group_distance_threshold': getFormValue('bullying_detection_group_distance_threshold', 'number'),
            'bullying_detection.group_min_size': getFormValue('bullying_detection_group_min_size', 'number'),
            'bullying_detection.rapid_movement_threshold': getFormValue('bullying_detection_rapid_movement_threshold', 'number'),

            // POSH detection settings
            'posh_detection.enabled': getFormValue('posh_detection_enabled', 'checkbox'),
            'posh_detection.confidence_threshold': getFormValue('posh_detection_confidence_threshold', 'number'),
            'posh_detection.isolation_distance_threshold': getFormValue('posh_detection_isolation_distance_threshold', 'number'),
            'posh_detection.proximity_threshold': getFormValue('posh_detection_proximity_threshold', 'number'),
            'posh_detection.prolonged_interaction_frames': getFormValue('posh_detection_prolonged_interaction_frames', 'number'),

            // Alert settings
            'alerts.save_to_database': getFormValue('alerts_save_to_database', 'checkbox'),
            'alerts.save_video_clips': getFormValue('alerts_save_video_clips', 'checkbox'),
            'alerts.clip_duration_seconds': getFormValue('alerts_clip_duration_seconds', 'number'),
            'alerts.max_alerts_per_minute': getFormValue('alerts_max_alerts_per_minute', 'number'),

            // Privacy settings
            'privacy.blur_faces': getFormValue('privacy_blur_faces', 'checkbox'),
            'privacy.save_only_pose_skeleton': getFormValue('privacy_save_only_pose_skeleton', 'checkbox'),
            'privacy.retention_days': getFormValue('privacy_retention_days', 'number')
        };

        // Remove undefined values
        Object.keys(updates).forEach(key => {
            if (updates[key] === undefined) {
                delete updates[key];
            }
        });

        // Send to API
        const response = await apiCall('/api/config/', {
            method: 'PUT',
            body: JSON.stringify({ updates })
        });

        if (response.status === 'success') {
            showToast(response.message, 'success');
            currentConfig = await apiCall('/api/config/');
        } else {
            showToast(`Validation failed: ${response.errors.join(', ')}`, 'error');
        }

    } catch (error) {
        console.error('Failed to save configuration:', error);
        showToast('Failed to save configuration', 'error');
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', initConfig);
