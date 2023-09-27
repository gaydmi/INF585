#pragma once

#include "boilerplate/base_exercise.hpp"

#ifdef SCENE

// Stores some parameters that can be set from the GUI
struct gui_scene_structure
{
    bool wireframe;
    bool texture_ship   = true;
    bool skybox = true;
    bool texture   = true;

};
struct particle_element
{
    vcl::vec3 p; // Position
    vcl::vec3 v; // Speed
};

struct scene_exercise : base_scene_exercise
{

    /** A part must define two functions that are called from the main function:
     * setup_data: called once to setup data before starting the animation loop
     * frame_draw: called at every displayed frame within the animation loop
     *
     * These two functions receive the following parameters
     * - shaders: A set of shaders.
     * - scene: Contains general common object to define the 3D scene. Contains in particular the camera.
     * - data: The part-specific data structure defined previously
     * - gui: The GUI structure allowing to create/display buttons to interact with the scene.
    */

    void setup_data(std::map<std::string,GLuint>& shaders, scene_structure& scene, gui_structure& gui);
    void frame_draw(std::map<std::string,GLuint>& shaders, scene_structure& scene, gui_structure& gui);


    // visual representation of a surface
    vcl::mesh_drawable sea;
     vcl::mesh sea_mesh;
    vcl::mesh_drawable cylinder;
    vcl::mesh_drawable cone;
    vcl::mesh_drawable surface_texture;
    gui_scene_structure gui_scene;
    GLuint texture_field_id;
    // Called every time the mouse is clicked

    // Data (p_i,t_i)
    std::vector<vcl::vec3> keyframe_position; // Given positions
    std::vector<float> keyframe_time;         // Time at given positions

    // Ship
    GLuint texture_ship;
    vcl::mesh_drawable_hierarchy ship;
    std::vector<vcl::vec3> ship_position;
    void display_ship(std::map<std::string,GLuint>& shaders, scene_structure& scene);

    // Skybox
    vcl::mesh_drawable skybox;
    GLuint texture_skybox;
    void display_skybox(std::map<std::string,GLuint>& shaders, scene_structure& scene);

    std::vector<std::vector<particle_element> > particles_flag, particles_sail_1, particles_sail_2, particles_sail_3;
    float L0;

    int N_flag = 15;
    vcl::mesh_drawable flag_surface;
    vcl::mesh surface_cpu_flag, surface_cpu_sail_1, surface_cpu_sail_2, surface_cpu_sail_3;
    GLuint texture_flag_id, texture_sail_id;
    vcl::timer_event timer;

};

#endif


