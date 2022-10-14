import Image from "next/future/image";
import ProfilePic from "../src/img/profilepic.jpg";
import MetaHeader from "../src/components/meta-header";

export default function Home() {
  return (
    <>
      <MetaHeader />
      <div className="h-screen">
        <div className="min-h-full bg-white px-4 py-16 sm:px-6 sm:py-24 grid place-items-center lg:px-8">
          <div className="mx-auto max-w-max">
            <main className="sm:flex">
              <Image
                src={ProfilePic}
                width={200}
                height={200}
                className="h-20 w-20 rounded-full"
                alt="Goh Shu Wei"
              />
              <div className="sm:ml-6">
                <div className="sm:border-l sm:border-gray-200 sm:pl-6">
                  <h1 className="text-4xl font-bold tracking-tight text-gray-900 sm:text-5xl">
                    GOH SW.
                  </h1>
                  <p className="mt-1 text-base text-gray-500">
                    There are no words to describe him
                  </p>
                </div>
                <div className="mt-6 flex space-x-3 sm:border-l sm:border-transparent sm:pl-6">
                  <a
                    href="https://twitter.com/gsw85"
                    target="_blank"
                    rel="noreferrer"
                  >
                    <div className="h-5 w-5 mr-2 text-twitter hover:text-sky-600" />
                  </a>
                  <a
                    href="https://github.com/gsw85"
                    target="_blank"
                    rel="noreferrer"
                  >
                    <div className="h-5 w-5 mr-2 text-black hover:text-gray-700" />
                  </a>
                  <a
                    href="https://my.linkedin.com/in/gsw85"
                    target="_blank"
                    rel="noreferrer"
                  >
                    <div className="h-5 w-5 mr-2 text-linkedin hover:text-cyan-900" />
                  </a>
                </div>
              </div>
            </main>
          </div>
        </div>
      </div>
    </>
  );
}
