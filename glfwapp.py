import sys
import platform
import OpenGL.GL as GL
from imgui_bundle import imgui,implot,ImVec2, ImVec4
# Always import glfw *after* imgui_bundle
# (since imgui_bundle will set the correct path where to look for the correct version of the glfw dynamic library)
import glfw
import ctypes

def glfw_error_callback(error: int, description: str):
    sys.stderr.write(f"Glfw Error {error}: {description}\n")

class GLFWapp:
    def __init__(self):
        # Setup window
        glfw.set_error_callback(glfw_error_callback)
        if not glfw.init():
            sys.exit(1)

        # Decide GL+GLSL versions
        if platform.system() == "Darwin":
            self.glsl_version = "#version 150"
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 2)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)  # // 3.2+ only
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
        else:
            # GL 3.0 + GLSL 130
            self.glsl_version = "#version 130"
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 2)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE) # // 3.2+ only
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)

        # Create window with graphics context
        self.window = glfw.create_window(1440, 768, "TorchPSC", None, None)
        if self.window is None:
            sys.exit(1)
        glfw.set_window_size_callback(self.window,self.on_window_resize)
        glfw.set_drop_callback(self.window,self.on_file_drop)
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)  # // Enable vsync

        imgui.create_context()
        io = imgui.get_io()
        io.config_flags |= imgui.ConfigFlags_.nav_enable_keyboard  # Enable Keyboard Controls
        io.config_flags |= imgui.ConfigFlags_.docking_enable  # Enable docking

        imgui.style_colors_dark()  ## imgui.style_colors_classic()

        # You need to transfer the window address to imgui.backends.glfw_init_for_opengl
        window_address = ctypes.cast(self.window, ctypes.c_void_p).value
        imgui.backends.glfw_init_for_opengl(window_address, True)
        imgui.backends.opengl3_init(self.glsl_version)

        self.init_imgui()

    def init_imgui(self):
        imgui.get_io().fonts.add_font_default()
        self.roboto = imgui.get_io().fonts.add_font_from_file_ttf(filename="./Roboto-Regular.ttf",size_pixels=14)

    def on_window_resize(self,window,w,h):
        pass

    def on_file_drop(self,window,filepath):
        print("parent : ",filepath)

    def run(self):
        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            imgui.backends.opengl3_new_frame()
            imgui.backends.glfw_new_frame()
            imgui.new_frame()
            #imgui.push_font(self.roboto )
            imgui.show_demo_window(True)
            imgui.pop_font()
            imgui.render()
            ##############################
            ##############################
            ## OpenGL Rendering
            ##
            display_w, display_h = glfw.get_framebuffer_size(self.window)
            GL.glViewport(0, 0, display_w, display_h)
            GL.glClearColor(0,0,0,0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)
            ##
            ## End opengl rendering
            ##############################
            ##############################
            imgui.backends.opengl3_render_draw_data(imgui.get_draw_data())
            glfw.swap_buffers(self.window)

    def cleanup(self):
        imgui.backends.opengl3_shutdown()
        imgui.backends.glfw_shutdown()
        imgui.destroy_context()
        glfw.destroy_window(self.window)
        glfw.terminate()