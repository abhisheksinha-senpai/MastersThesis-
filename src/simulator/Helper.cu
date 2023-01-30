#include "Helper.cuh"

unsigned int SCR_WIDTH = 800;
unsigned int SCR_HEIGHT = 600;
float deltaTime = 0.0f;
float lastFrame = 0.0f;
float camradius = 7.0f;
float cameraspeed = 0.02f;
float camX = camradius;
float camY = 0.0f;
float camZ = 0.0f;
bool firstMouse = true;
float yaw = -90.0f;
float pitch = 0.0f;
float lastX = SCR_WIDTH / 2.0;
float lastY = SCR_HEIGHT / 2.0;
float fov = 45.0f;

glm::vec3 cameraPos = glm::vec3(camX, camY, camZ);
glm::vec3 cameraFront = glm::vec3(-1.0f, 0.0f, 0.0f);
glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);

glm::mat4 view = glm::mat4(1.0f);
glm::mat4 model = glm::mat4(1.0f);
glm::mat4 proj = glm::mat4(1.0f);


__host__ void mouse_callback(GLFWwindow* window, double xposIn, double yposIn);
__host__ void framebuffer_size_callback(GLFWwindow* window, int width, int height);

__host__ void display_init(GLFWwindow** window)
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    (*window) = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);

    if (*window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return;
    }
    glfwMakeContextCurrent(*window);
    glfwSetFramebufferSizeCallback(*window, framebuffer_size_callback);
    glfwSetCursorPosCallback(*window, mouse_callback);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return;
    }
    stbi_set_flip_vertically_on_load(true);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glfwSetInputMode(*window, GLFW_CURSOR, GLFW_CURSOR_NORMAL );
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable( GL_BLEND );
    printf("%s %s %s\n", glGetString(GL_VERSION), glGetString(GL_VENDOR), glGetString(GL_RENDERER));
    printf("Display initialized.....\n");
}

__host__ void model_init(ResourceManager &r_manager, Shader &ourShader, Model &ourModel, int NX, int NY, int NZ,
glm::f32vec3 scale, glm::f32vec3 origin)
{
    r_manager.load_shader("resources/shaders/vertex/model_shader.vs", "VERTEX", ourShader.vertex_shader);
    r_manager.load_shader("resources/shaders/fragment/model_shader.fs", "FRAGMENT", ourShader.fragment_shader);
    ourShader.create_vs_shader(ourShader.vertex_shader.c_str());
    ourShader.create_fs_shader(ourShader.fragment_shader.c_str());
    ourShader.compile();
    std::string model_name = "resources/BlenderModels/Cube_8cuts.obj";
    ourModel = Model((char *)model_name.c_str(), scale, origin);
    printf("Model initialized.....\n");
}

void domain_init(int NX, int NY, int NZ,
                float **rho, float **ux, float **uy,float **uz)
{
    int sz = NX*NY*NZ*sizeof(float);
    *rho = (float *)malloc(sz);
    *ux =  (float *)malloc(sz);
    *uy =  (float *)malloc(sz);
    *uz =  (float *)malloc(sz);

    memset(*rho, 0, sz);
    memset(*ux, 0, sz);
    memset(*uy, 0, sz);
    memset(*uz, 0, sz);
    
    int loc = 0, X1, Y1, Z1;
    for(int j=0;j<NY;j++)
    {
        for(int i=0;i<NX;i++)
        {
            for(int k=0;k<NZ;k++)
            {
                loc = i+j*NX+k*NX*NY;
                if(i == 0 || j == 0 || k == 0 || i == NX-1 || j == NY-1 || k == NZ-1)
                {
                    (*rho)[loc] = 99999.0f;
                    (*ux)[loc] = 0.0f;
                    (*uy)[loc] = 0.0f;
                    (*uz)[loc] = 0.0f;
                }
                else
                {
                    // if(j==3*NY/4 && j<8*NY/9 && i<8*NX/9 && i>NX/4 && k<8*NZ/9 && k>NZ/4)
                    // if(j>7*NY/9 && j<8*NY/9)
                    if(j<1*NY/20)
                        (*rho)[loc] = 1.0f;
                    // if(powf((i-NX/2), 2.0f)+powf((j-5*NY/8), 2.0f)+powf((k-NZ/2), 2.0f)<powf(NX/16, 2.0f))
                    //     (*rho)[loc] = 1.0f;
                    // else if((j<NY*1/4))// && (i>NX/4 && i<3*NX/4) && (k<3*NZ/4 && k>NZ/4))
                    //     (*rho)[loc] = 1.0f;
                    else
                        (*rho)[loc] = 0.001f;
                    (*ux)[loc] = 0.0f;
                    (*uy)[loc] = 0.0f;
                    (*uz)[loc] = 0.0f;
                }
            }
        }
    }
    printf("Domain initialized...\n");
}

__host__ void scene_init(float *rho_gpu, float *ux_gpu, float *uy_gpu, float *uz_gpu,
                         float *rho, float *ux, float *uy, float *uz, 
                         int NX, int NY, int NZ)
{
    int sz = NX*NY*NZ*sizeof(float);
    checkCudaErrors(cudaMemcpy(temp_cell_type_gpu, rho, sz, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(ux_gpu, ux, sz, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(uy_gpu, uy, sz, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(uz_gpu, uz, sz, cudaMemcpyHostToDevice));
    printf("Scene initialized.....\n");
}

__host__ void scene_cleanup(Vertex **nodeLists, Vertex **nodeData, int *vertex_size_per_mesh,
                            float *rho, float *ux, float *uy, float *uz)
{
    free(rho);
    free(ux);
    free(uy);
    free(uz);
    free(nodeLists);
    free(nodeData);
    free(vertex_size_per_mesh);
}

__host__ void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

__host__ void processInput(GLFWwindow* window)
{
    float currentFrame = glfwGetTime();
    deltaTime = currentFrame - lastFrame;
    lastFrame = currentFrame;

    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    const float cameraSpeed = 5.0f * deltaTime; // adjust accordingly
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        cameraPos += cameraSpeed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        cameraPos -= cameraSpeed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
}

__host__ void mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
{
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
    {
        float xpos = static_cast<float>(xposIn);
        float ypos = static_cast<float>(yposIn);

        if (firstMouse)
        {
            lastX = xpos;
            lastY = ypos;
            firstMouse = false;
        }

        float xoffset = xpos - lastX;
        float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top
        lastX = xpos;
        lastY = ypos;

        float sensitivity = 0.25f; // change this value to your liking
        xoffset *= sensitivity;
        yoffset *= sensitivity;

        yaw += xoffset;
        pitch += yoffset;

        // make sure that when pitch is out of bounds, screen doesn't get flipped
        if (pitch > 89.0f)
            pitch = 89.0f;
        if (pitch < -89.0f)
            pitch = -89.0f;

        glm::vec3 front;
        front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        front.y = sin(glm::radians(pitch));
        front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));

        cameraFront = glm::normalize(front);
    }
    else if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_RELEASE)
    {
        lastX = SCR_WIDTH / 2.0;
        lastY = SCR_HEIGHT / 2.0;
        firstMouse = true;
    }
}

__host__ void draw_model( GLFWwindow* window, Shader& shader, Model& objmodel, glm::f32vec3 scale)
{
    shader.use();
    // view/projection transformations
    view = glm::lookAt(cameraPos, cameraPos+cameraFront, cameraUp);
    proj = glm::perspective(glm::radians(45.0f), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 1000.0f);

    glUniformMatrix4fv(glGetUniformLocation(shader.get_shader_pgm(), "view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(shader.get_shader_pgm(), "projection"), 1, GL_FALSE, glm::value_ptr(proj));
    // render the loaded model
    objmodel.Draw(shader, scale);
    model = glm::mat4(1);
}

__host__ void transfer_fluid_data(float *rho, float*ux, float *uy,float *uz,
                                  float *rho_gpu, float *ux_gpu, float*uy_gpu, float* uz_gpu, 
                                  int NX, int NY, int NZ)
{
    int sz = NX*NY*NZ*sizeof(float);
    cudaMemcpy(rho, (void *)mass_gpu, sz, cudaMemcpyDeviceToHost);
    cudaMemcpy(ux, (void *)Fx_gpu, sz, cudaMemcpyDeviceToHost);
    cudaMemcpy(uy, (void *)Fy_gpu, sz, cudaMemcpyDeviceToHost);
    cudaMemcpy(uz, (void *)Fz_gpu, sz, cudaMemcpyDeviceToHost);
}

__host__ void draw_fluid(float *rho, float*ux, float *uy, float *uz,
                         float *rho_gpu, float *ux_gpu, float*uy_gpu, float* uz_gpu,
                         int NX, int NY, int NZ, 
                         ParticleSystem &fluid, glm::f32vec3 model_scale, glm::f32vec3 dis_scale)
{
    transfer_fluid_data(rho, ux, uy, uz,
                        mass_gpu, ux_gpu, uy_gpu, uz_gpu,
                        NX, NY, NZ);
    
    fluid.update_particles(NX, NY, NZ, rho, ux, uy, uz, model_scale);
    fluid.draw_particles(SCR_WIDTH, SCR_HEIGHT, cameraPos, cameraFront, cameraUp, dis_scale);
}
int n = 0;
__host__ void display ( float *rho, float*ux, float *uy, float *uz,
                        float *rho_gpu, float *ux_gpu, float*uy_gpu, float* uz_gpu,
                        int NX, int NY, int NZ, 
                        ParticleSystem &fluid, glm::f32vec3 mod_scale, glm::f32vec3 dis_scale,
                        GLFWwindow** window, Shader& shader, Model &model, Geometry &fluidDomain)
{
    glClearColor(0.35f, 0.15f, 0.35f, 0.05f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    draw_model( *window, shader, model, dis_scale);

    fluidDomain.draw_geometry(SCR_WIDTH, SCR_HEIGHT, cameraPos, cameraFront, cameraUp);
    
    draw_fluid(rho, ux, uy,uz, mass_gpu, ux_gpu, uy_gpu, uz_gpu, NX, NY, NZ, fluid, mod_scale, dis_scale);

    processInput(*window);
    glfwPollEvents();
    glfwSwapBuffers(*window);
}