all: ClearScreen AsczEngine Run

ClearScreen:
	rm -f AsczEngine.exe \
	clear

AsczEngine:
	nvcc \
		-I include/Default \
		-I include/MathLib \
		-I include/Graphic3D \
		-I include/Shader3D \
		\
		-I libraries/SFML/include \
		-L libraries/SFML/lib \
		\
		-o AsczEngine \
		\
		src/MathLib/Vector.cu \
		src/MathLib/Matrix.cu \
		src/Graphic3D/Mesh3D.cu \
		src/Graphic3D/Camera3D.cu \
		src/Shader3D/Render3D.cu \
		src/Shader3D/Lighting3D.cu \
		src/Shader3D/Buffer3D.cu \
		src/Shader3D/SFMLTexture.cu \
		src/Default/FpsHandler.cu \
		src/Default/CsLogHandler.cu \
		\
		AsczEngine.cu \
		\
		-lsfml-system \
		-lsfml-window \
		-lsfml-graphics \
		-lsfml-audio \
		-lopenal32 \
		\
		-rdc=true \
		--expt-relaxed-constexpr \
		--extended-lambda \

Run:
	./AsczEngine

clean:
	rm -f AsczEngine.exe

# Add <-mwindows> so when you run AsczEngine.exe
# it doesnt open a terminal
# (unless you need debugging and stuff ofc)