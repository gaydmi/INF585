
#include "scene.hpp"
#include <vector>
#include <random>
#include <algorithm>
#ifdef SCENE

// Add vcl namespace within the current one - Allows to use function from vcl library without explicitely preceeding their name with vcl::
using namespace vcl;


// Flag relevant parts
static void set_gui(timer_basic& timer);
mesh create_flag(int N, float L0);
std::vector<std::vector<particle_element> > particles_flag, particles_sail_1, particles_sail_2, particles_sail_3;
mesh create_grid(int N, float L0, std::vector<std::vector<particle_element> >& particles);
void update_positions_flag(int N, float L0, float t, float dt, float m, float K, float mu, vec3 g, vec3 translation, mesh& surface_cpu_flag,  std::vector<std::vector<particle_element> >&  particles);
void update_positions_sail(int N, float L0, float t, float dt, float m, float K, float mu, vec3 g, vec3 translation, mesh& surface_cpu_sail,  std::vector<std::vector<particle_element> >&  particles);

// Perlin sea relevant parts
float evaluate_sea_z(float u, float v);
vec3 evaluate_sea(float u, float v, float t);
mesh create_sea();
void update_sea(mesh& sea, float t);
mesh sea_mesh;
int N = 100;

// Ship relevant parts
mesh_drawable_hierarchy create_ship(float radius, const vec3& p0);
mesh create_hull(float radius, float a, float b, float c, const vec3& p0);
vec3 update_ship_position(float t);

vcl::mesh create_skybox();



/** This function is called before the beginning of the animation loop
    It is used to initialize all part-specific data */
void scene_exercise::setup_data(std::map<std::string,GLuint>& , scene_structure& scene, gui_structure& )
{
    // Create visual sea surface
    sea_mesh = create_sea();

    texture_field_id = texture_gpu( image_load_png("data/sea.png") );
    // Setup initial camera mode and position
    scene.camera.camera_type = camera_control_spherical_coordinates;
    scene.camera.scale = 10.0f;
    scene.camera.apply_rotation(0,0,0,1.2f);


    mesh surface_cpu;
    surface_cpu.position     = {{-0.2f,0,0}, { 0.2f,0,0}, { 0.2f, 0.4f,0}, {-0.2f, 0.4f,0}};
    surface_cpu.texture_uv   = {{0,1}, {1,1}, {1,0}, {0,0}};
    surface_cpu.connectivity = {{0,1,2}, {0,2,3}};

    surface_texture = surface_cpu;
    surface_texture.uniform_parameter.shading = {1,0,0}; // set pure ambiant component (no diffuse, no specular) - allow to only see the color of the texture


    timer.scale = 0.5f;



    skybox = create_skybox();
    skybox.uniform_parameter.shading = {1,0,0};
    skybox.uniform_parameter.rotation = rotation_from_axis_angle_mat3({1,0,0},-3.014f/2.0f);
    texture_skybox = texture_gpu(image_load_png("data/skybox.png"));

    L0 = 0.225f;
    surface_cpu_flag = create_grid(N_flag, L0/2.0f, particles_flag);
    texture_flag_id = texture_gpu( image_load_png("data/flag.png") );
    texture_sail_id = texture_gpu( image_load_png("data/sail.png") );
    surface_cpu_sail_1 = create_grid(N_flag*2, L0, particles_sail_1);
    surface_cpu_sail_2 = create_grid(N_flag*3, L0, particles_sail_2);
    surface_cpu_sail_3 = create_grid(N_flag*2, L0, particles_sail_3);
    ship = create_ship(0.5f, {0, 0, 2});

}



/** This function is called at each frame of the animation loop.
    It is used to compute time-varying argument and perform data data drawing */
void scene_exercise::frame_draw(std::map<std::string,GLuint>& shaders, scene_structure& scene, gui_structure& )
{
    timer.update();
    set_gui(timer);

    float t = timer.t;

    for(size_t ku=0; ku<N; ++ku)
    {
        for(size_t kv=0; kv<N; ++kv)
        {
            const float u = ku/(N-1.0f);
            const float v = kv/(N-1.0f);

            sea_mesh.position[kv+N*ku] = evaluate_sea(u, v, t);
            

        }
    }
    sea = sea_mesh;


    // Simulation time step (dt)
    float dt = timer.scale*0.01f;

    // Simulation parameters
    const float m  = 0.01f;        // particle mass
    const float K  = 5.0f;         // spring stiffness
    const float mu_flag = 0.093f;       // damping coefficient
    const float mu_sail = 0.01f;       // damping coefficient


    const vec3 g_flag   = {5.0f*(std::sin( timer.t*1.8f)+0.7f),-1.51f*(std::cos( timer.t*2.5f)+1.0f),-4.21f*(std::cos( timer.t*3.0f))}; // gravity
    const vec3 g_sail  = {(std::sin( timer.t*4.8f)),-1.41f*(std::cos( timer.t*2.5f)+0.6f), (std::cos( timer.t*3.2f))}; // gravity
    const vec3 translation_sail_1 = vec3({-3.25f, -17.5f, 27});
    const vec3 translation_sail_2 = vec3({-5, -10.5f, 31});
    const vec3 translation_sail_3 = vec3({-3.25f, -3.5f, 27});
    const vec3 translation_flag = vec3({0, -10.5f, 34.5f});
    update_positions_flag(N_flag, L0/2.0f, t, dt, m, K, mu_flag, g_flag, translation_flag, surface_cpu_flag, particles_flag);
    update_positions_sail(N_flag*2, L0/1.1f, t, dt, m, K, mu_sail, g_sail, translation_sail_1, surface_cpu_sail_1, particles_sail_1);
    update_positions_sail(N_flag*3, L0/1.1f, t, dt, m, K, mu_sail, g_sail, translation_sail_2, surface_cpu_sail_2, particles_sail_2);
    update_positions_sail(N_flag*2, L0/1.1f, t, dt, m, K, mu_sail, g_sail, translation_sail_3, surface_cpu_sail_3, particles_sail_3);


    glEnable( GL_POLYGON_OFFSET_FILL ); // avoids z-fighting when displaying wireframe
    glBindTexture(GL_TEXTURE_2D, texture_flag_id);


    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);

    // Display flag
    glPolygonOffset( 1.0, 1.0 );
    flag_surface = surface_cpu_flag;
    flag_surface.draw(shaders["mesh"], scene.camera);

    glBindTexture(GL_TEXTURE_2D, texture_sail_id);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);


    glPolygonOffset( 1.0, 1.0 );

    // Display sails
    flag_surface = surface_cpu_sail_1;
    flag_surface.uniform_parameter.color = {1.0f, 1.0f, 1.0f};
    flag_surface.draw(shaders["mesh"], scene.camera);

    flag_surface = surface_cpu_sail_2;
    flag_surface.draw(shaders["mesh"], scene.camera);


    flag_surface = surface_cpu_sail_3;
    flag_surface.draw(shaders["mesh"], scene.camera);

    glBindTexture(GL_TEXTURE_2D, texture_field_id);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);

    // Display ship
    glPolygonOffset( 1.0, 1.0 );
    ship.translation("hull") = update_ship_position(t);

    sea.draw(shaders["mesh"], scene.camera);

    display_ship(shaders, scene);
    display_skybox(shaders, scene);



}

void scene_exercise::display_skybox(std::map<std::string,GLuint>& shaders, scene_structure& scene)
{
    if(gui_scene.skybox)
    {
        if(gui_scene.texture)
        glBindTexture(GL_TEXTURE_2D,texture_skybox);
        skybox.uniform_parameter.scaling = 150.0f;
        skybox.uniform_parameter.translation = scene.camera.camera_position() + vec3(0,0,-50.0f);
        skybox.draw(shaders["mesh"], scene.camera);
        glBindTexture(GL_TEXTURE_2D,scene.texture_white);
    }
}

void scene_exercise::display_ship(std::map<std::string,GLuint>& shaders, scene_structure& scene)
{

    const size_t N = ship_position.size();
        
    ship.draw(shaders["mesh"], scene.camera);
    if(gui_scene.texture_ship)
        glBindTexture(GL_TEXTURE_2D, texture_ship);

    if( gui_scene.wireframe ){
        for(size_t k=0; k<N; ++k)
        {
            const vec3& p = ship_position[k];
            ship.draw(shaders["wireframe"], scene.camera);
        }
    }
}

// Evaluate height of the sea for any (u,v) \in [0,1]
float evaluate_sea_z(float u, float v)
{
    int size = 4;
    const std::vector<vec2> u_vec {
            {0.0f, 0.0f}, {0.5f,0.5f},
            {0.2f,0.7f}, {0.8f,0.7f}};
    const std::vector<float> h_vec {3.0f, -1.5f, 2.0f, -2.0f};
    const std::vector<float> sigma_vec {0.4f, 0.15f, 2.2f, 0.2f};

    float z = 0.0f;
    for(int i=0; i<size; i++){

        const float d_i = norm(vec2(u,v)-u_vec[i])/sigma_vec[i];

        z+=h_vec[i]*std::exp(-d_i*d_i);
    }

    return z;
}

// Evaluate 3D position of the sea for any (u,v) \in [0,1]
vec3 evaluate_sea(float u, float v, float t)
{


    const float scaling = 0.5f;
    const int octave = 7;
    const float persistency = 0.8f;
    const float height = 10.9f;
    // Evaluate Perlin noise
    const float noise = perlin(scaling*u, scaling*v+ t/50.0f, octave, persistency );

    const float x = 200*(u-0.5f);
    const float y = 200*(v-0.5f);
    const float z = height*(noise*0.2f);//+evaluate_sea_z(u,v));

    return {x,y,z-12};
}

// Generate sea mesh
mesh create_sea()
{
    // Number of samples of the sea is N x N
    mesh sea; // temporary sea storage (CPU only)
    sea.position.resize(N*N);
    sea.texture_uv.resize(N*N);
    // Fill sea geometry
    for(size_t ku=0; ku<N; ++ku)
    {
        for(size_t kv=0; kv<N; ++kv)
        {
            // Compute local parametric coordinates (u,v) \in [0,1]
            const float u = ku/(N-1.0f);
            const float v = kv/(N-1.0f);

            // Compute coordinates
            sea.position[kv+N*ku] = evaluate_sea(u,v, 0);
            sea.texture_uv[kv+N*ku] = {u, v};
        }
    }

    // Generate triangle organization
    //  Parametric surface with uniform grid sampling: generate 2 triangles for each grid cell
    const unsigned int Ns = N;
    for(unsigned int ku=0; ku<Ns-1; ++ku)
    {
        for(unsigned int kv=0; kv<Ns-1; ++kv)
        {
            const unsigned int idx = kv + N*ku; // current vertex offset

            const index3 triangle_1 = {idx, idx+1+Ns, idx+1};
            const index3 triangle_2 = {idx, idx+Ns, idx+1+Ns};

            sea.connectivity.push_back(triangle_1);
            sea.connectivity.push_back(triangle_2);

        }
    }
    return sea;
}


void update_sea(mesh& sea, float t){
    for(size_t ku=0; ku<N; ++ku)
    {
        for(size_t kv=0; kv<N; ++kv)
        {
            const float u = ku/(N-1.0f);
            const float v = kv/(N-1.0f);
            sea.position[kv+N*ku] = evaluate_sea(u,v,t);
        }
    }
}


vcl::mesh create_skybox()
{
    const vec3 p000 = {-1,-1,-1};
    const vec3 p001 = {-1,-1, 1};
    const vec3 p010 = {-1, 1,-1};
    const vec3 p011 = {-1, 1, 1};
    const vec3 p100 = { 1,-1,-1};
    const vec3 p101 = { 1,-1, 1};
    const vec3 p110 = { 1, 1,-1};
    const vec3 p111 = { 1, 1, 1};

    mesh skybox;

    skybox.position = {
        p000, p100, p110, p010,
        p010, p110, p111, p011,
        p100, p110, p111, p101,
        p000, p001, p010, p011,
        p001, p101, p111, p011,
        p000, p100, p101, p001
    };


    skybox.connectivity = {
        {0,1,2}, {0,2,3}, {4,5,6}, {4,6,7},
        {8,11,10}, {8,10,9}, {17,16,19}, {17,19,18},
        {23,22,21}, {23,21,20}, {13,12,14}, {13,14,15}
    };

    const float e = 1e-3f;
    const float u0 = 0.0f;
    const float u1 = 0.25f+e;
    const float u2 = 0.5f-e;
    const float u3 = 0.75f-e;
    const float u4 = 1.0f;
    const float v0 = 0.0f;
    const float v1 = 1.0f/3.0f+e;
    const float v2 = 2.0f/3.0f-e;
    const float v3 = 1.0f;
    skybox.texture_uv = {
        {u1,v1}, {u2,v1}, {u2,v2}, {u1,v2},
        {u1,v2}, {u2,v2}, {u2,v3}, {u1,v3},
        {u2,v1}, {u2,v2}, {u3,v2}, {u3,v1},
        {u1,v1}, {u0,v1}, {u1,v2}, {u0,v2},
        {u4,v1}, {u3,v1}, {u3,v2}, {u4,v2},
        {u1,v1}, {u2,v1}, {u2,v0}, {u1,v0}
    };


    return skybox;

}

static void set_gui(timer_basic& timer)
{
    // Can set the speed of the animation
    float scale_min = 0.05f;
    float scale_max = 2.0f;
    ImGui::SliderScalar("Time scale", ImGuiDataType_Float, &timer.scale, &scale_min, &scale_max, "%.2f s");

    // Start and stop animation
    if (ImGui::Button("Stop"))
        timer.stop();
    if (ImGui::Button("Start"))
        timer.start();

}


mesh create_hull(float radius, float a, float b, float c, const vec3& p0) 
{
    mesh m;

    // Number of samples
    const size_t Nu = 20;
    const size_t Nv = 20;
    
    // Geometry
    size_t idx = 0;
    std::vector<size_t> indices;

    for( size_t ku=0; ku<Nu; ++ku ) {
        for( size_t kv=0; kv<Nv+1; ++kv ) {

            const float u = static_cast<float>(ku)/static_cast<float>(Nu);
            const float v = static_cast<float>(kv)/static_cast<float>(Nv);

            const float theta = static_cast<float>( 0.5*3.14159f*u+3.14159f); 
            const float phi   = static_cast<float>( 1.2*3.14159f*v -0.1 *3.14159f);

            const float x = a * radius * std::sin(theta) * std::cos(phi);
            const float y = b * radius * std::sin(theta) * std::sin(phi);
            const float z = c * radius * std::cos(theta);
            const vec3 p = {x,y,z};
            //const vec3 n = normalize(p);
            indices.push_back(idx);   
            m.position.push_back( p+p0 );
            idx++;
        }
    }

    // Connectivity 
    for(size_t ku=0; ku<Nu; ++ku) {
        for( size_t kv=0; kv<Nv; ++kv) {

            const unsigned int u00 = static_cast<unsigned int>( kv + Nv*ku );
            const unsigned int u10 = static_cast<unsigned int>( kv+1 + Nv*ku );
            const unsigned int u01 = static_cast<unsigned int>( kv + Nv*(ku+1) );
            const unsigned int u11 = static_cast<unsigned int>( kv+1 + Nv*(ku+1) );

            unsigned int posu00 = static_cast<unsigned int>( std::find(indices.begin(), indices.end(), u00) - indices.begin());
            unsigned int posu01 = static_cast<unsigned int>( std::find(indices.begin(), indices.end(), u01) - indices.begin()); 
            unsigned int posu10 = static_cast<unsigned int>( std::find(indices.begin(), indices.end(), u10) - indices.begin()); 
            unsigned int posu11 = static_cast<unsigned int>( std::find(indices.begin(), indices.end(), u11) - indices.begin());                    
            
            if (std::find(indices.begin(), indices.end(), u00) != indices.end() &&
                std::find(indices.begin(), indices.end(), u10) != indices.end() &&
                std::find(indices.begin(), indices.end(), u11) != indices.end()) {
                    m.connectivity.push_back({posu00, posu10, posu11});
            }

            if (std::find(indices.begin(), indices.end(), u00) != indices.end() &&
                std::find(indices.begin(), indices.end(), u01) != indices.end() &&
                std::find(indices.begin(), indices.end(), u11) != indices.end()) {
                    m.connectivity.push_back({posu11, posu01, posu00});
            }
        }
    }
    
    for( size_t ku=0; ku<Nu-1; ++ku ) {

            const unsigned int u00 = static_cast<unsigned int>( Nv-1 + Nv*ku );
            const unsigned int u10 = static_cast<unsigned int>( 0 + Nv*ku );
            const unsigned int u01 = static_cast<unsigned int>( Nv-1 + Nv*(ku+1) );
            const unsigned int u11 = static_cast<unsigned int>( 0 + Nv*(ku+1) );
            

            unsigned int posu00 = static_cast<unsigned int>( std::find(indices.begin(), indices.end(), u00) - indices.begin());
            unsigned int posu01 = static_cast<unsigned int>( std::find(indices.begin(), indices.end(), u01) - indices.begin()); 
            unsigned int posu10 = static_cast<unsigned int>( std::find(indices.begin(), indices.end(), u10) - indices.begin()); 
            unsigned int posu11 = static_cast<unsigned int>( std::find(indices.begin(), indices.end(), u11) - indices.begin());                    
            
            if (std::find(indices.begin(), indices.end(), u00) != indices.end() &&
                std::find(indices.begin(), indices.end(), u10) != indices.end() &&
                std::find(indices.begin(), indices.end(), u11) != indices.end()) {
                    m.connectivity.push_back({posu00, posu10, posu11});
            }

            if (std::find(indices.begin(), indices.end(), u00) != indices.end() &&
                std::find(indices.begin(), indices.end(), u01) != indices.end() &&
                std::find(indices.begin(), indices.end(), u11) != indices.end()) {
                    m.connectivity.push_back({posu11, posu01, posu00});
            }
            
    }
    
    // Deck connectivity
    size_t height = 17;
    for( size_t kv=0; kv<Nv-1; ++kv ) {

        const unsigned int u00 = static_cast<unsigned int>( kv + height*Nv);
            const unsigned int u10 = static_cast<unsigned int>( kv + 1 + height*Nv );
            const unsigned int u01 = static_cast<unsigned int>( Nv-kv + height*Nv );
            const unsigned int u11 = static_cast<unsigned int>( Nv-kv-1 + height*Nv );
            
            

            unsigned int posu00 = static_cast<unsigned int>( std::find(indices.begin(), indices.end(), u00) - indices.begin());
            unsigned int posu01 = static_cast<unsigned int>( std::find(indices.begin(), indices.end(), u01) - indices.begin()); 
            unsigned int posu10 = static_cast<unsigned int>( std::find(indices.begin(), indices.end(), u10) - indices.begin()); 
            unsigned int posu11 = static_cast<unsigned int>( std::find(indices.begin(), indices.end(), u11) - indices.begin());                    
        
            if (std::find(indices.begin(), indices.end(), u00) != indices.end() &&
                std::find(indices.begin(), indices.end(), u10) != indices.end() &&
                std::find(indices.begin(), indices.end(), u11) != indices.end()) {
                    m.connectivity.push_back({posu00, posu10, posu11});
            }

            if (std::find(indices.begin(), indices.end(), u00) != indices.end() &&
                std::find(indices.begin(), indices.end(), u01) != indices.end() &&
                std::find(indices.begin(), indices.end(), u11) != indices.end()) {
                    m.connectivity.push_back({posu11, posu01, posu00});
            }
    }

    return m;
}

mesh_drawable_hierarchy create_ship(float radius, const vec3& p0)  {
    mesh_drawable_hierarchy hierarchy;

    const float a = 0.8f;
    const float b = 4.5f;
    const float c = 0.5f;


    hierarchy.scaling = 10.0f;

    // Hull
    mesh hull = create_hull(radius, a, b, c, p0);
    hierarchy.add_element(hull, "hull","root", {0,0,0});
    hierarchy.mesh_visual("hull").uniform_parameter.color = {0.4f, 0.3f, 0.3f};

    // Masts
    mesh ship_mast = mesh_primitive_cylinder(0.01f, {0, -0.35f, 1.8f}, {0, -0.35f, 3.0f}, 20, 20);
    hierarchy.add_element(ship_mast,"mast","hull", {0.0f, 0.0f , 0.1f});
    hierarchy.mesh_visual("mast").uniform_parameter.color = {0.4f, 0.3f, 0.3f};

    mesh ship_mast2 = mesh_primitive_cylinder(0.01f, {-0.35f, -0.35f, 2.4f}, {0.35f, -0.35f, 2.4f}, 20, 20);
    hierarchy.add_element(ship_mast2,"mast2","mast", {0.0f, 0.0f , 0.2f});
    hierarchy.mesh_visual("mast2").uniform_parameter.color = {0.4f, 0.3f, 0.3f};

    mesh ship_mast3 = mesh_primitive_cylinder(0.01f, {0, -1.05f, 1.8f}, {0, -1.05f, 3.4f}, 20, 20);
    hierarchy.add_element(ship_mast3,"mast3","hull", {0.0f, 0.0f , 0.1f});
    hierarchy.mesh_visual("mast3").uniform_parameter.color = {0.4f, 0.3f, 0.3f};

    mesh ship_mast4 = mesh_primitive_cylinder(0.01f, {-0.5f, -1.05f, 2.8f}, {0.5f, -1.05f, 2.8f}, 20, 20);
    hierarchy.add_element(ship_mast4,"mast4","mast3", {0.0f, 0.0f , 0.2f});
    hierarchy.mesh_visual("mast4").uniform_parameter.color = {0.4f, 0.3f, 0.3f};

    mesh ship_cyl = mesh_primitive_cylinder(0.06f, {0, -1.05f, 2.9f}, {0, -1.05f, 2.97f}, 20, 20);
    hierarchy.add_element(ship_cyl,"cyl","mast3", {0.0f, 0.0f , 0.1f});
    hierarchy.mesh_visual("cyl").uniform_parameter.color = {0.4f, 0.3f, 0.3f};

    mesh ship_mast5 = mesh_primitive_cylinder(0.01f, {0, -1.75f, 1.8f}, {0, -1.75f, 3.0f}, 20, 20);
    hierarchy.add_element(ship_mast5,"mast5","hull", {0.0f, 0.0f , 0.1f});
    hierarchy.mesh_visual("mast5").uniform_parameter.color = {0.4f, 0.3f, 0.3f};

    mesh ship_mast6 = mesh_primitive_cylinder(0.01f, {-0.35f, -1.75f, 2.4f}, {0.35f, -1.75f, 2.4f}, 20, 20);
    hierarchy.add_element(ship_mast6,"mast6","mast5", {0.0f, 0.0f , 0.2f});
    hierarchy.mesh_visual("mast6").uniform_parameter.color = {0.4f, 0.3f, 0.3f};

    // Superstructure
    mesh ship_quad =  mesh_primitive_parallelepiped(
     {-0.3f, -0.05f, 1.9f}, {0, 0 , 0.25f},
     {0.6, 0, 0}, {0, 0.6, 0});
    hierarchy.add_element(ship_quad, "quad","hull", {0.0f, 0.0f , 0.05f});
    hierarchy.mesh_visual("quad").uniform_parameter.color = {0.4f, 0.3f, 0.3f};

    return hierarchy;


}

vec3 update_ship_position(float t)
{
        const float u = 0.5f;
        const float v = 0.5f;
        const float vx = 0.08f;
        const float vy = 0.45;
        const vec3 p = evaluate_sea(u, v, t) - vec3({vx * t, vy * t, 17.65f});

        return p;
}

/** Compute spring force applied on particle pi from particle pj */
vec3 spring_force(const vec3& pi, const vec3& pj, float L0, float K)
{
    vec3 u = {pi.x-pj.x, pi.y-pj.y, pi.z-pj.z};
    float L = pow(u.x*u.x+u.y*u.y+u.z*u.z, 0.5);
    u = u/L;
    return -K*(L-L0)*u;
}
mesh create_grid(int N, float L0, std::vector<std::vector<particle_element> >& particles){
    mesh surface_cpu;

    for (int i = 0; i < N; i++) {
        particle_element part;
        part.p = {i*L0, 0, 0};     // Initial position of particle A
        part.v = {0, 0, 0};     // Initial speed of particle A
        particles.push_back({part});
        for (int j = 1; j < N; j++) {
            particle_element part;
            part.p = {i*L0, 0, -j*L0};     // Initial position of particle A
            part.v = {0, 0, 0};     // Initial speed of particle A
            particles[i].push_back(part);
        }
    }

    surface_cpu.position.resize(N*N);
    surface_cpu.texture_uv.resize(N*N);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {

            const float u = i/(N-1.0f);
            const float v = j/(N-1.0f);
            surface_cpu.position[j+N*i] = particles[i][j].p;
            surface_cpu.texture_uv[j+N*i] = {u, v};
            const unsigned int idx = j + N*i; // current vertex offset
            if (i<N-1 and j<N-1){
                const index3 triangle_1 = {idx, idx+1+N, idx+1};
                const index3 triangle_2 = {idx, idx+N, idx+1+N};
                surface_cpu.connectivity.push_back(triangle_1);
                surface_cpu.connectivity.push_back(triangle_2);
            }

        }
    }
    return surface_cpu;
}


void update_positions_flag(int N, float L0, float t, float dt, float m, float K, float mu, vec3 g, vec3 translation, mesh& surface_cpu,  std::vector<std::vector<particle_element> >&  particles)
{

    std::vector<std::vector<vec3> > forces;
    for (int i = 1; i < N; i++) {
        forces.push_back({vec3 {0,0,0}});
        for (int j = 0; j < N; j++) {

            particle_element part_prec_bottom = particles[i-1][j];

            particle_element part_curr = particles[i][j];

            // Forces
            const vec3 f_spring_c_b  = spring_force(part_curr.p, part_prec_bottom.p, L0, K);
            const vec3 f_weightB  = m*g;
            const vec3 f_dampingB = -mu*part_curr.v;
            const vec3 FB = f_spring_c_b+f_weightB+f_dampingB;
            vec3 add = {0,0,0};
            if (j>0){
                particle_element part_prec_left = particles[i][j-1];
                add += spring_force(part_curr.p, part_prec_left.p, L0, K);
            }
            if (i<N-1){
                if (j<N-1){
                    particle_element part_next_right = particles[i][j+1];
                    add += spring_force(part_curr.p, part_next_right.p, L0, K);
                }
                particle_element part_next_top = particles[i+1][j];

                add += spring_force(part_curr.p, part_next_top.p, L0, K);
            }
            if ((i>=N-2) and (j>=N-2)){
                add=-FB+forces[i-1][j-1];
            }
            forces[i-1].push_back(FB+add);
        }
    }



    for (int i = 0; i < N; i++) {
         for (int j = 0; j < N; j++) {

            particle_element part_curr = particles[i][j];
            // Numerical Integration (Verlet)
            if (i>0){

                vec3& posB = part_curr.p; // position of particle
                vec3& vitB = part_curr.v; // speed of particle

                vitB = vitB + dt*forces[i-1][j+1]/m;
                posB = posB + dt*vitB;
                particles[i][j].p = posB;

                particles[i][j].v = vitB;
                if ((i==N-1) and (j==N-1)){

                    particles[i][j].v = vitB;
                }
            }
            surface_cpu.position[j+N*i] = particles[i][j].p+update_ship_position(t)+translation;
        }

    }
}


void update_positions_sail(int N, float L0, float t, float dt, float m, float K, float mu, vec3 g, vec3 translation, mesh& surface_cpu,  std::vector<std::vector<particle_element> >&  particles){

    std::vector<std::vector<vec3> > forces;
    for (int i = 0; i < N; i++) {
        forces.push_back({vec3 {0,0,0}});
        for (int j = 0; j < N; j++) {
            particle_element part_curr = particles[i][j];

            // Forces

            const vec3 f_weightB  = m*g;
            const vec3 f_dampingB = -mu*part_curr.v;
            const vec3 FB = f_weightB+f_dampingB;
            vec3 add = {0,0,0};
            if (i>0){
                particle_element part_prec_bottom = particles[i-1][j];
                add += spring_force(part_curr.p, part_prec_bottom.p, L0, K);
            }
            if (j>0){
                particle_element part_prec_left = particles[i][j-1];
                add += spring_force(part_curr.p, part_prec_left.p, L0, K);
            }
            if (i<N-1){
                particle_element part_next_top = particles[i+1][j];
                add += spring_force(part_curr.p, part_next_top.p, L0, K);
            }
            if (j<N-1){
                particle_element part_next_right = particles[i][j+1];
                add += spring_force(part_curr.p, part_next_right.p, L0, K);
            }
            if ((i==N-1 and j==N-1) or (i==0 and j==N-1) or (i==N-1 and j==0) or (i%(int(N/5))==0 and j==0)) add = -FB;
            forces[i].push_back(FB+add);
        }
    }



    for (int i = 0; i < N; i++) {
         for (int j = 0; j < N; j++) {
            particle_element part_curr = particles[i][j];
            {
                vec3& posB = part_curr.p; // position of particle
                vec3& vitB = part_curr.v; // speed of particle

                vitB = vitB + dt*forces[i][j+1]/m;
                posB = posB + dt*vitB;
                particles[i][j].p = posB;

                particles[i][j].v = vitB;
                if ((i==N-1) and (j==N-1)){

                    particles[i][j].v = vitB;
                }
            }
            surface_cpu.position[j+N*i] = particles[i][j].p+update_ship_position(t)+translation;

        }

    }
}


#endif

