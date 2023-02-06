#include "Domain.hpp"

void Geometry::draw_geometry(int SCR_WIDTH, int SCR_HEIGHT, glm::vec3 cameraPos, glm::vec3 cameraFront, glm::vec3 cameraUp)
{
    domainShader.use();
    glm::mat4 model = glm::mat4(1);
    model = glm::translate(model, br);
    model = glm::scale(model, br);

    glm::mat4 view = glm::lookAt(cameraPos, cameraPos+cameraFront, cameraUp); 
    glm::mat4 proj = glm::perspective(glm::radians(45.0f), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 1000.0f);
    glUniformMatrix4fv(glGetUniformLocation(domainShader.get_shader_pgm(), "view_domain"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(domainShader.get_shader_pgm(), "model_domain"), 1, GL_FALSE, glm::value_ptr(model));
    glUniformMatrix4fv(glGetUniformLocation(domainShader.get_shader_pgm(), "projection_domain"), 1, GL_FALSE, glm::value_ptr(proj));

    // glGenVertexArrays(1, &VAO);
    // glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);           // for vertex data
    glBufferData(GL_ARRAY_BUFFER, 252 * sizeof(float), body, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7*sizeof(float), (void*)0);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 7*sizeof(float), (void*)12);

    glBindVertexArray(VAO);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glDrawArrays(GL_TRIANGLES, 0, 36);
	glBindVertexArray(0);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    // glDeleteBuffers(1, &VBO);
    // glDeleteVertexArrays(1, &VAO);
}

Geometry::Geometry(glm::f32vec3 scale): br(scale)
{
    ResourceManager r_manager = ResourceManager();
    r_manager.load_shader("resources/shaders/vertex/domain_shader.vs", "VERTEX", domainShader.vertex_shader);
    r_manager.load_shader("resources/shaders/fragment/domain_shader.fs", "FRAGMENT", domainShader.fragment_shader);
    domainShader.create_vs_shader(domainShader.vertex_shader.c_str());
    domainShader.create_fs_shader(domainShader.fragment_shader.c_str());
    domainShader.compile();
    domainShader.use();
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);           // for vertex data
    glBufferData(GL_ARRAY_BUFFER, 252 * sizeof(float), body, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7*sizeof(float), (void*)0);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 7*sizeof(float), (void*)12);
    glBindVertexArray(0);
}

Geometry::~Geometry()
{
    glDeleteBuffers(1, &VBO);
    glDeleteVertexArrays(1, &VAO);
}